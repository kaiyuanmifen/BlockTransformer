''' Define the Layers '''
import torch.nn as nn
import torch
from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward


__author__ = "DL"

class EncoderLayer_block(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1,N_blocks=4,N_slots=128):
        super(EncoderLayer_block, self).__init__()

        self.ax_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)  
        self.axslf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)  
              
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.cls_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)

        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.pos_ffn_ax = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
                    
            
        self.N_blocks=N_blocks
        self.N_slots=N_slots
        self.d_model=d_model
        self.auxilary_slots_query=torch.nn.Parameter(torch.randn((N_blocks,N_slots,d_model)))

    def forward(self, enc_input, slf_attn_mask=None):
        ###squence len changes over batch 



        #input size bz, T ,Embz
        T=enc_input.shape[1]
        bz=enc_input.shape[0]
        blockSize=T//(self.N_blocks-1)

        ##auxilary slot read from each position
        all_auxilary_slots=[]
        for i in range(self.N_blocks):
            if i<(self.N_blocks-1):
                PositionsInBlock=enc_input[:,(i*blockSize+1):((i+1)*blockSize+1),:]#the first token is for classification, therefore, not in block
            else:
                PositionsInBlock=enc_input[:,(i*blockSize+1):,:]


            auxilary_query=self.auxilary_slots_query[i,:,:].unsqueeze(0).repeat(bz,1,1)

            auxilary_slot,_=self.ax_attn(auxilary_query, PositionsInBlock, PositionsInBlock)
            all_auxilary_slots.append(auxilary_slot)

        all_auxilary_slots=torch.cat(all_auxilary_slots,1)#(bz,N_blocks*N_slots,d_v)

        ############communication among axilary slots
        all_auxilary_slots,_=self.axslf_attn(all_auxilary_slots, all_auxilary_slots, all_auxilary_slots)
        
        all_auxilary_slots=self.pos_ffn_ax(all_auxilary_slots)

        all_auxilary_slots=all_auxilary_slots.reshape(bz,self.N_blocks,self.N_slots,self.d_model)

        #self attention within each block
        All_block_positions=[]

        ######all tokens but the first one only attend to auxillary slots

        for i in range(self.N_blocks):
            if i<(self.N_blocks-1):
                PositionsInBlock=enc_input[:,(i*blockSize+1):((i+1)*blockSize+1),:]
            else:
                PositionsInBlock=enc_input[:,(i*blockSize+1):,:]

            auxilary_slot=all_auxilary_slots[:,i,:,:]

            PositionsInBlock,_= self.slf_attn(PositionsInBlock,auxilary_slot, 
                                                               auxilary_slot)
            #PositionsInBlock_and_axiSlot=torch.cat([PositionsInBlock,auxilary_slot],1)

           # PositionsInBlock_and_axiSlot,_= self.slf_attn(PositionsInBlock_and_axiSlot, 
           #                                                     PositionsInBlock_and_axiSlot, 
            #                                                    PositionsInBlock_and_axiSlot)
                                                            
            #PositionsInBlock=PositionsInBlock_and_axiSlot[:,:PositionsInBlock.shape[1],:]

            All_block_positions.append(PositionsInBlock)



        All_block_positions=torch.cat(All_block_positions,1)

        ###the first token is for classification , therefore attend to all positions

        CLSToken=enc_input[:,0,:].unsqueeze(1)


        CLSToken,enc_slf_attn=self.cls_attn(CLSToken,
            all_auxilary_slots.reshape(all_auxilary_slots.shape[0],all_auxilary_slots.shape[1]*all_auxilary_slots.shape[2],all_auxilary_slots.shape[3]),
            all_auxilary_slots.reshape(all_auxilary_slots.shape[0],all_auxilary_slots.shape[1]*all_auxilary_slots.shape[2],all_auxilary_slots.shape[3]))



        enc_output=torch.cat([CLSToken,All_block_positions],1)

        ###positionwise ffl 

        enc_output = self.pos_ffn(enc_output)
        
        return enc_output, enc_slf_attn

class EncoderLayer_hierachy(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1,N_blocks=4,N_slots=128):
        super(EncoderLayer_hierachy, self).__init__()

        self.ax_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)  

        self.gws_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)  

        self.gwsSelf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)  
        
        
        self.ax_GWS_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)  


        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)

        self.cls_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

        self.pos_ffn_GWS = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.pos_ffn_ax = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
                    


        self.N_blocks=N_blocks
        self.N_slots=N_slots
        self.d_model=d_model
        self.auxilary_slots_query=torch.nn.Parameter(torch.randn((N_blocks,N_slots,d_model)))
        self.GWS_slots_query=torch.nn.Parameter(torch.randn((1,N_slots,d_model)))

    def forward(self, enc_input, slf_attn_mask=None):
        ###squence len changes over batch 

        #input size bz, T ,Embz
        T=enc_input.shape[1]
        bz=enc_input.shape[0]
        blockSize=T//(self.N_blocks-1)


        ##auxilary slot read from each position
        all_auxilary_slots=[]
        for i in range(self.N_blocks):
            if i<(self.N_blocks-1):
                PositionsInBlock=enc_input[:,(i*blockSize+1):((i+1)*blockSize+1),:]
            else:
                PositionsInBlock=enc_input[:,(i*blockSize+1):,:]

            auxilary_query=self.auxilary_slots_query[i,:,:].unsqueeze(0).repeat(bz,1,1)

            auxilary_slot,_=self.ax_attn(auxilary_query, PositionsInBlock, PositionsInBlock)
            all_auxilary_slots.append(auxilary_slot)

        all_auxilary_slots=torch.cat(all_auxilary_slots,1)#(bz,N_blocks*N_slots,d_v)
        all_auxilary_slots=self.pos_ffn_ax(all_auxilary_slots)
        ####GWS slots read from all auxilary slots



        GWS_slots,_=self.gws_attn(self.GWS_slots_query.repeat(bz,1,1),all_auxilary_slots,all_auxilary_slots)


        ############communication among GWS slots
        GWS_slots,_=self.gwsSelf_attn(GWS_slots, GWS_slots, GWS_slots)

        GWS_slots=self.pos_ffn_GWS(GWS_slots)

        
        ###auxilary slots read from GWS_slots


        all_auxilary_slots,_=self.ax_GWS_attn(all_auxilary_slots,GWS_slots,GWS_slots)

        all_auxilary_slots=all_auxilary_slots.reshape(bz,self.N_blocks,self.N_slots,self.d_model)

        #self attention within each block
        All_block_positions=[]
        for i in range(self.N_blocks):
            if i<(self.N_blocks-1):
                PositionsInBlock=enc_input[:,(i*blockSize+1):((i+1)*blockSize+1),:]
            else:
                PositionsInBlock=enc_input[:,(i*blockSize+1):,:]

            auxilary_slot=all_auxilary_slots[:,i,:,:]

            #PositionsInBlock_and_axiSlot=torch.cat([PositionsInBlock,auxilary_slot],1)

            # PositionsInBlock_and_axiSlot,_= self.slf_attn(PositionsInBlock_and_axiSlot, 
            #                                                     PositionsInBlock_and_axiSlot, 
            #                                                     PositionsInBlock_and_axiSlot)

            #PositionsInBlock=PositionsInBlock_and_axiSlot[:,:PositionsInBlock.shape[1],:]

            PositionsInBlock,_= self.slf_attn(PositionsInBlock,auxilary_slot,auxilary_slot)
                                                            
            
            All_block_positions.append(PositionsInBlock)



        All_block_positions=torch.cat(All_block_positions,1)

        ###the first token is for classification , therefore attend to all positions

        CLSToken=enc_input[:,0,:].unsqueeze(1)
        CLSToken,enc_slf_attn=self.cls_attn(CLSToken,GWS_slots,GWS_slots)



        enc_output=torch.cat([CLSToken,All_block_positions],1)

        ###positionwise ffl 

        enc_output = self.pos_ffn(enc_output)
        
        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(
            self, dec_input, enc_output,
            slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn
