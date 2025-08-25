#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import collections
import torch
import numpy as np
import transformers
import pickle
from os.path import join
import data.bert_finetuning_er_seq2seq_dataset as module_data
import model.bert_binding as module_arch
from parse_config import ConfigParser
import pandas as pd


def test(model ,data_loader, antibody_tokenizer, antigen_tokenizer,device,config):
    model.eval()
    result_dict = {'heavy': [], 
                    'light':[], 
                    'antigen':[],
                    'y_pred': []}
    with torch.no_grad():
        for batch_idx, (antibody_a_tokenized,antibody_b_tokenized, receptor_tokenized) in enumerate(data_loader):
            antibody_a_tokenized = {k: v.to(device) for k, v in antibody_a_tokenized.items()}
            antibody_b_tokenized = {k: v.to(device) for k, v in antibody_b_tokenized.items()}
            receptor_tokenized = {k: v.to(device) for k, v in receptor_tokenized.items()}
            output = model(antibody_a_tokenized, antibody_b_tokenized, receptor_tokenized, device)
            
            y_pred = output
            
            y_pred = y_pred.cpu().detach().numpy()
            result_dict['y_pred'].append(y_pred)

            antibody = antibody_tokenizer.batch_decode(antibody_a_tokenized['input_ids'],
                                                    skip_special_tokens=True)
            antibody = [s.replace(" ", "") for s in antibody]

            light = antibody_tokenizer.batch_decode(antibody_b_tokenized['input_ids'],
                                                    skip_special_tokens=True)
            light = [s.replace(" ", "") for s in light]


            receptor = antigen_tokenizer.batch_decode(receptor_tokenized['input_ids'],
                                                    skip_special_tokens=True)
            receptor = [s.replace(" ", "") for s in receptor]
            result_dict['heavy'].append(antibody)
            result_dict['light'].append(light)
            result_dict['antigen'].append(receptor)


    y_pred = np.concatenate(result_dict['y_pred'])



    test_df = pd.DataFrame({'heavy': [v for l in result_dict['heavy'] for v in l],
                            'light': [v for l in result_dict['light'] for v in l],
                            'antigen': [v for l in result_dict['antigen'] for v in l],
                            'y_pred': list(y_pred.flatten())})
    test_df.to_csv(join(config.log_dir, 'test_result.csv'), index=False)

    return test_df, join(config.log_dir, 'test_result.csv')




def main(config):
    logger = config.get_logger('eval_generation')

    # fix random seeds for reproducibility
    seed = config['data_loader']['args']['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    # setup data_loader instances
    config['data_loader']['args']['logger'] = logger
    data_loader = config.init_obj('data_loader', module_data)

    antibody_tokenizer = data_loader.get_antibody_tokenizer()
    antigen_tokenizer = data_loader.get_antigen_tokenizer()

    # build model architecture, then print to console

    model = config.init_obj('arch', module_arch)
    logger.info('Loading checkpint from {}'.format(
        config['discriminator_resume']))
    checkpoint = torch.load(config['discriminator_resume'])
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict, strict=False)
    model.to("cuda")


    """Test."""
    logger = config.get_logger('test')
    test_df, csv_path = test(model=model, data_loader=data_loader, antibody_tokenizer=antibody_tokenizer, antigen_tokenizer=antigen_tokenizer, device="cuda", config=config)
    print(csv_path)
    return csv_path


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='./config/common/bert_eval_generation.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-local_rank', '--local_rank', default=None, type=str,
                      help='local rank for nGPUs training')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
