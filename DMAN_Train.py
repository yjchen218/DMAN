import logging
import argparse
import math
import os
import sys
import random
import numpy

from sklearn import metrics
from time import strftime, localtime
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from transformers import BertModel

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split

from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, ABSADataset
from models import LSTM, IAN, MemNet, RAM, TD_LSTM, TC_LSTM, Cabasc, ATAE_LSTM, TNet_LF, AOA, MGAN, ASGCN, LCF_BERT, ASGCN_BERT
from models.aen import CrossEntropyLoss_LSR, AEN_BERT, AEN_GLOVE
from models.bert_spc import BERT_SPC
from models.supconloss import SupConLoss
from models.selfsupconloss import SelfSupConLoss
from my_model.ig_bert import Integrated_Gradients, IG_BERT, GraphConvolution, IG_GCN

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Instructor:
    def __init__(self, opt):
        self.opt = opt
        self.best_model_path = ' '
        self.IG_GCN = IG_GCN(opt).to(opt.device)

        if 'bert' in opt.model_name:
            tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
            bert = BertModel.from_pretrained("./bert_model")
            self.model = opt.model_class(bert, opt).to(opt.device)
        else:
            tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
                max_seq_len=opt.max_seq_len,
                dat_fname='{0}_tokenizer.dat'.format(opt.dataset))
            embedding_matrix = build_embedding_matrix(
                word2idx=tokenizer.word2idx,
                embed_dim=opt.embed_dim,
                dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))
            self.model = opt.model_class(embedding_matrix, opt).to(opt.device)

        self.trainset = ABSADataset(opt.dataset_file['train'], tokenizer)
        self.testset = ABSADataset(opt.dataset_file['test'], tokenizer)
        assert 0 <= opt.valset_ratio < 1
        if opt.valset_ratio > 0:
            valset_len = int(len(self.trainset) * opt.valset_ratio)
            self.trainset, self.valset = random_split(self.trainset, (len(self.trainset)-valset_len, valset_len))
        else:
            self.valset = self.testset

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('> n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel: 
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)


    def _train_stage1(self, criterion, optimizer, scheduler, train_data_loader, val_data_loader):
        logger.info("\n" + "="*30 + " STAGE 1: Training Entire IG_BERT " + "="*30)
        max_val_acc = 0
        max_val_f1 = 0
        max_val_epoch = 0
        global_step = 0
        path = None
        
        # 确保所有参数开启梯度
        self.model.train() 
        for param in self.model.parameters():
            param.requires_grad = True

        for i_epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('Stage 1 - epoch: {}'.format(i_epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            
            self.model.train()
            
            for i_batch, batch in enumerate(train_data_loader):
                global_step += 1
                optimizer.zero_grad()

                inputs = [batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                targets = batch['polarity'].to(self.opt.device)
                
                batch_size = len(batch['polarity'])
                saliency_dummy = [torch.ones(batch_size, self.opt.max_seq_len).to(self.opt.device),
                                  torch.ones(batch_size, self.opt.max_seq_len).to(self.opt.device)]

                if self.opt.model_name == 'ig_bert':
                    outputs_bert, _ = self.model(inputs, saliency_dummy)
                    loss = criterion(outputs_bert, targets)
                else:
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    outputs_bert = outputs 

                loss.backward()
                optimizer.step()
                scheduler.step()
                
                n_correct += (torch.argmax(outputs_bert, -1) == targets).sum().item()
                n_total += len(outputs_bert)
                loss_total += loss.item() * len(outputs_bert)
                
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))
            
            val_acc, val_f1, _, _ = self._evaluate_acc_f1(val_data_loader, stage1_mode=True)
            logger.info('> val_acc_bert: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))

            if val_acc > max_val_acc:
                max_val_acc = val_acc
                max_val_epoch = i_epoch
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                path = 'state_dict/{0}_{1}_val_acc_{2}'.format(self.opt.model_name, self.opt.dataset, round(val_acc, 4))
                torch.save(self.model.state_dict(), path)
                logger.info('>> saved: {}'.format(path))
            
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1
            if i_epoch - max_val_epoch >= self.opt.patience:
                print('>> early stop.')
                break

        return path

    def _train_stage2(self, criterion, optimizer_total, scheduler_total, train_data_loader, val_data_loader, IG):
        logger.info("\n" + "="*30 + " STAGE 2: Training GCN + IG_BERT Custom Layers " + "="*30)
        max_val_acc = 0
        max_val_f1 = 0
        max_val_epoch = 0
        global_step = 0
        path = None
        gcn_path = None


        self.model.train() 
        self.IG_GCN.train()

        self.model.bert.eval() 
        for param in self.model.bert.parameters():
            param.requires_grad = False

        for i_epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('Stage 2 - epoch: {}'.format(i_epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            
            for i_batch, batch in enumerate(train_data_loader):
                global_step += 1
                optimizer_total.zero_grad()
                batch_size = len(batch['polarity'])

                inputs = [batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                targets = batch['polarity'].to(self.opt.device)
                adj = batch['dependency_graph'].to(self.opt.device)
                
                saliency_dummy = [torch.ones(batch_size, self.opt.max_seq_len).to(self.opt.device),
                                  torch.ones(batch_size, self.opt.max_seq_len).to(self.opt.device)]

                if self.opt.model_name == 'ig_bert':
                    with torch.enable_grad():
                        outputs_bert, f_states = self.model(inputs, saliency_dummy)
                        loss_bert = criterion(outputs_bert, targets)
                        loss_bert.backward(retain_graph=True) 

                        saliency_1 = IG(loss_bert, f_states[2]) * batch_size / 2
                        saliency_2 = IG(loss_bert, f_states[3]) * batch_size / 2
                        
                    saliency = [saliency_1.detach(), saliency_2.detach()]
                    text_embedding = f_states[4].detach()
                    
                    if i_batch == 0:
                        logger.info('> attention: %s', torch.mean(f_states[3][0], dim=1)[1:10]/0.2)
                        logger.info('> saliency_1: %s ', saliency_1[0][1:10]/2.5)
                        logger.info('> saliency_1: %s ', saliency_1[1][1:10]/2.5)

                    outputs = self.IG_GCN(text_embedding, saliency, adj, inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    
                    optimizer_total.step()
                    scheduler_total.step()
                else:
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

            val_acc, val_f1, _, _ = self._evaluate_acc_f1(val_data_loader, stage1_mode=False, criterion=criterion, IG=IG)
            logger.info('> val_acc_gcn: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))
            
            if val_acc > max_val_acc:
                base_dir = 'pretrained_ig_bert/{0}/state_dict'.format(self.opt.dataset)
                if not os.path.exists(base_dir):
                    os.makedirs(base_dir)
                
                path = '{0}/{1}_{2}_val_acc_{3}'.format(base_dir, self.opt.model_name, self.opt.dataset, round(val_acc, 4))
                torch.save(self.model.state_dict(), path)    
                logger.info('>> saved BERT: {}'.format(path))

                gcn_path = '{0}/ig_gcn_{1}_val_acc_{2}'.format(base_dir, self.opt.dataset, round(val_acc, 4))
                torch.save(self.IG_GCN.state_dict(), gcn_path)
                logger.info('>> saved GCN: {}'.format(gcn_path))
                
                max_val_acc = val_acc
                max_val_epoch = i_epoch
                    
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1
            if i_epoch - max_val_epoch >= self.opt.patience:
                print('>> early stop.')
                break
        
        return path, gcn_path

    def _evaluate_acc_f1(self, data_loader, stage1_mode=True, criterion=None, IG=None, t=False):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        
        self.model.eval()
        self.IG_GCN.eval()
        
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                t_inputs = [t_batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_batch['polarity'].to(self.opt.device)
                t_adj = t_batch['dependency_graph'].to(self.opt.device)
                batch_size = len(t_batch['polarity'])
                
                saliency_dummy = [torch.ones(batch_size, self.opt.max_seq_len).to(self.opt.device),
                                  torch.ones(batch_size, self.opt.max_seq_len).to(self.opt.device)]

                if self.opt.model_name == 'ig_bert':
                    if stage1_mode:
                        t_outputs, _ = self.model(t_inputs, saliency_dummy)
                    else:
                        with torch.enable_grad():
                            t_outputs_bert, f_states = self.model(t_inputs, saliency_dummy)
                            loss_bert = criterion(t_outputs_bert, t_targets)
                            loss_bert.backward(retain_graph=True)
                            
                            saliency_1 = IG(loss_bert, f_states[2]) * batch_size / 2
                            saliency_2 = IG(loss_bert, f_states[3]) * batch_size / 2
                            saliency = [saliency_1, saliency_2]
                            text_embedding = f_states[3]
                        
                        t_outputs = self.IG_GCN(text_embedding, saliency, t_adj, t_inputs)
                else:
                    t_outputs = self.model(t_inputs)

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)
                
                if t and not stage1_mode:
                    incorrect = (torch.argmax(t_outputs, -1) != t_targets)
                    if incorrect.any():
                        incorrect_texts = [t_inputs[0][i] for i in range(len(incorrect)) if incorrect[i]]
                        tokenizer111 = BertTokenizer.from_pretrained("./bert_model")
                        special_chars = {'[CLS]', '[SEP]', '[PAD]', '[#]'}
                        for text in incorrect_texts:
                            tokens = tokenizer111.convert_ids_to_tokens(text)
                            clean_list = [item for item in tokens if item not in special_chars]
                            logger.info("预测错误的句子:{}".format(' '.join(clean_list)))

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        return acc, f1, 0, 0

    def run(self):
        # Loss and Utilities
        criterion = CrossEntropyLoss_LSR(self.opt.device)
        IG = Integrated_Gradients(self.opt.top_k)
        
        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)
        
        logger.info("\n" + "#"*50 + "\n STARTING STAGE 1: IG_BERT Full Fine-tuning \n" + "#"*50)
        
        self._reset_params()
        
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.lr, weight_decay=self.opt.l2reg)
        
        eval_steps = int(len(train_data_loader))
        t_total = int(eval_steps * self.opt.num_epoch)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)

        if self.opt.model_name == 'ig_bert':
            best_bert_path = self._train_stage1(criterion, optimizer, scheduler, train_data_loader, val_data_loader)
        else:            
            self._train_stage1(criterion, optimizer, scheduler, train_data_loader, val_data_loader)
            return

        logger.info("Loading best IG_BERT weights from Stage 1...")
        best_bert_path = "pretrained_ig_bert/{0}/ig_bert_{1}".format(self.opt.dataset, self.opt.dataset)
        self.model.load_state_dict(torch.load(best_bert_path))
        
        logger.info("\n" + "#"*50 + "\n STARTING STAGE 2: GCN + IG_BERT Head Training \n" + "#"*50)
        params_to_optimize = [
            {'params': filter(lambda p: p.requires_grad, self.model.parameters())}, # IG_BERT 中剩下来 requires_grad=True 的部分
            {'params': self.IG_GCN.parameters()}
        ]
        
        optimizer_total = self.opt.optimizer(params_to_optimize, lr=self.opt.lr, weight_decay=self.opt.l2reg)
        scheduler_total = get_linear_schedule_with_warmup(optimizer_total, num_warmup_steps=0, num_training_steps=t_total)
        
        best_bert_path_s2, best_gcn_path_s2 = self._train_stage2(criterion, optimizer_total, scheduler_total,
                                                                 train_data_loader, val_data_loader, IG)
        
        logger.info("Loading best weights from Stage 2 for Final Testing...")
        self.model.load_state_dict(torch.load(best_bert_path_s2))
        self.IG_GCN.load_state_dict(torch.load(best_gcn_path_s2))
        
        test_acc, test_f1, _, _ = self._evaluate_acc_f1(test_data_loader, stage1_mode=False, criterion=criterion, IG=IG, t=True)
        logger.info('>> Final test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert_spc', type=str)
    parser.add_argument('--dataset', default='laptop', type=str, help='twitter, restaurant, laptop')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--lr', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--num_epoch', default=20, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=16, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=85, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=1234, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0, type=float, help='set ratio between 0 and 1 for validation support')
    parser.add_argument('--local_context_focus', default='cdm', type=str, help='local context focus mode, cdw or cdm')
    parser.add_argument('--SRD', default=3, type=int, help='semantic-relative-distance, see the paper of LCF-BERT model')
    parser.add_argument('--top_k', default=300, type=int)
    opt = parser.parse_args()

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(opt.seed)

    model_classes = {
        'lstm': LSTM,
        'td_lstm': TD_LSTM,
        'tc_lstm': TC_LSTM,
        'atae_lstm': ATAE_LSTM,
        'ian': IAN,
        'memnet': MemNet,
        'ram': RAM,
        'cabasc': Cabasc,
        'tnet_lf': TNet_LF,
        'aoa': AOA,
        'mgan': MGAN,
        'asgcn': ASGCN,
        'bert_spc': BERT_SPC,
        'aen_bert': AEN_BERT,
        'aen_glove': AEN_GLOVE,
        'lcf_bert': LCF_BERT,
        'asgcn_bert': ASGCN_BERT,
        'ig_bert': IG_BERT
    }
    dataset_files = {
        'twitter': {
            'train': './datasets/acl-14-short-data/train.raw',
            'test': './datasets/acl-14-short-data/test.raw'
        },
        'restaurant': {
            'train': './datasets/semeval14/Restaurants_Train.xml.seg',
            'test': './datasets/semeval14/Restaurants_Test_Gold.xml.seg'
        },
        'laptop': {
            'train': './datasets/semeval14/Laptops_Train.xml.seg',
            'test': './datasets/semeval14/Laptops_Test_Gold.xml.seg'
        },
        'restaurant15': {
            'train': './datasets/semeval15/Restaurants_Train.xml.seg',
            'test': './datasets/semeval15/Restaurants_Test_Gold.xml.seg'
        },
        'restaurant16': {
            'train': './datasets/semeval16/Restaurants_Train.xml.seg',
            'test': './datasets/semeval16/Restaurants_Test_Gold.xml.seg'
        },
        'mams': {
            'train': './datasets/mams/Mams_Train.xml.seg',
            'test': './datasets/mams/Mams_Test_Gold.xml.seg'
        },     
    }
    input_colses = {
        'lstm': ['text_indices'],
        'td_lstm': ['left_with_aspect_indices', 'right_with_aspect_indices'],
        'tc_lstm': ['left_with_aspect_indices', 'right_with_aspect_indices', 'aspect_indices'],
        'atae_lstm': ['text_indices', 'aspect_indices'],
        'ian': ['text_indices', 'aspect_indices'],
        'memnet': ['context_indices', 'aspect_indices'],
        'ram': ['text_indices', 'aspect_indices', 'left_indices'],
        'cabasc': ['text_indices', 'aspect_indices', 'left_with_aspect_indices', 'right_with_aspect_indices'],
        'tnet_lf': ['text_indices', 'aspect_indices', 'aspect_boundary'],
        'aoa': ['text_indices', 'aspect_indices'],
        'mgan': ['text_indices', 'aspect_indices', 'left_indices'],
        'asgcn': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph'],
        'bert_spc': ['concat_bert_indices', 'concat_segments_indices'],
        'aen_bert': ['text_bert_indices', 'aspect_bert_indices'],
        'aen_glove': ['text_indices', 'aspect_indices'],
        'lcf_bert': ['concat_bert_indices', 'concat_segments_indices', 'text_bert_indices', 'aspect_bert_indices'],
        'asgcn_bert': ['text_bert_indices', 'aspect_bert_indices', 'left_bert_indices', 'dependency_graph',
                        'concat_bert_indices', 'concat_segments_indices'],
        'ig_bert': ['concat_bert_indices', 'concat_segments_indices', 'text_bert_indices', 'aspect_bert_indices', 'left_bert_indices']
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
        'adamw': torch.optim.AdamW
    }
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    log_file = '{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()
