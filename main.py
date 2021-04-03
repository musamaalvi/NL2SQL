import sys
sys.path.insert(0, 'RoBERTaNL2SQL')


from RoBERTaNL2SQL import load_data
import torch
import json,argparse
from RoBERTaNL2SQL import load_model
from RoBERTaNL2SQL import roberta_training
from RoBERTaNL2SQL import corenlp_local
from RoBERTaNL2SQL import seq2sql_model_testing
from RoBERTaNL2SQL import seq2sql_model_training_functions
from RoBERTaNL2SQL import model_save_and_infer
from RoBERTaNL2SQL import dev_function
from RoBERTaNL2SQL import infer_functions
import time
import os
import nltk

from dbengine_sqlnet import DBEngine
from torchsummary import summary
from tqdm.notebook import tqdm
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize


def print_result(epoch, acc, dname):
    ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx, acc_x = acc

    print(f'{dname} results ------------')
    print(
        f" Epoch: {epoch}, ave loss: {ave_loss}, acc_sc: {acc_sc:.3f}, acc_sa: {acc_sa:.3f}, acc_wn: {acc_wn:.3f}, \
        acc_wc: {acc_wc:.3f}, acc_wo: {acc_wo:.3f}, acc_wvi: {acc_wvi:.3f}, acc_wv: {acc_wv:.3f}, acc_lx: {acc_lx:.3f}, acc_x: {acc_x:.3f}"
    )
path_wikisql = "E:/Masters/Semester4/Thesis/Implementations/RobertaGithub/NL2SQL"
BATCH_SIZE = 8

train_data, train_table, dev_data, dev_table, train_loader, dev_loader = load_data.get_data(path_wikisql, batch_size = BATCH_SIZE)
test_data,test_table,test_loader = load_data.get_test_data(path_wikisql, batch_size = BATCH_SIZE)
zero_data,zero_table,zero_loader = load_data.get_zero_data(path_wikisql, batch_size = BATCH_SIZE)    # Data to test Zero Shot Learning


roberta_model, tokenizer, configuration = load_model.get_roberta_model()          # Loads the RoBERTa Model
seq2sql_model = load_model.get_seq2sql_model(configuration.hidden_size)


model_optimizer, roberta_optimizer = load_model.get_optimizers(seq2sql_model , roberta_model)



EPOCHS = 30
acc_lx_t_best = 0.40            # Creats checkpoint so that a worse model does not get saved
epoch_best = 0   
def run():
                    
    for epoch in range(0, EPOCHS):
        acc_train = dev_function.train( seq2sql_model, roberta_model, model_optimizer, roberta_optimizer, tokenizer, configuration, path_wikisql, train_loader)
        acc_dev, results_dev, cnt_list = dev_function.test(seq2sql_model, roberta_model, model_optimizer, tokenizer, configuration, path_wikisql, dev_loader, mode="dev")
        print_result(epoch, acc_train, 'train')
        print_result(epoch, acc_dev, 'dev')
        acc_lx_t = acc_dev[-2]
        if acc_lx_t > acc_lx_t_best:                  # IMPORTANT : Comment out this whole if block if you are using a shortcut to the original
            acc_lx_t_best = acc_lx_t                  #             Drive Folder, otherwise an error will stop the execution of the code.
            epoch_best = epoch                        #             You cannot edit the files in the original folder
                                                    #             Download and Upload a separate copy to change the files.
            
            # save best model
            state = {'model': seq2sql_model.state_dict()}
            if os.path.isdir(os.path.join(path_wikisql, str(epoch))):
                torch.save(state, os.path.join(path_wikisql, str(epoch) , 'model_best.pt'))
            else:
                os.mkdir(os.path.join(path_wikisql, str(epoch)))
                torch.save(state, os.path.join(path_wikisql, str(epoch) , 'model_best.pt'))

            state = {'model_roberta': roberta_model.state_dict()}
            if os.path.isdir(os.path.join(path_wikisql, str(epoch))):
                torch.save(state, os.path.join(path_wikisql, str(epoch) , 'model_roberta_best.pt'))
            else:
                os.mkdir(os.path.join(path_wikisql, str(epoch)))
                torch.save(state, os.path.join(path_wikisql, str(epoch) , 'model_roberta_best.pt'))
        print(f" Best Dev lx acc: {acc_lx_t_best} at epoch: {epoch_best}")
if __name__ == "__main__":
    print("usama")
    run()
    