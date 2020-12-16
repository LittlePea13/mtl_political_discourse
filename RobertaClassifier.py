from transformers import RobertaModel
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import copy
import math


class BertForPoliticalClassification(nn.Module):
    def __init__(self, language, cache_dir, num_labels, num_polit, attention_dropout, fc_dropout):
        super(BertForPoliticalClassification, self).__init__()
        self.num_labels = num_labels
        #self.bert_weights = BertModel(config)
        self.temp_bert = RobertaModel.from_pretrained(
                'roberta-base', cache_dir=cache_dir, num_labels=num_labels
            )
        self.bert = AdaptedBertModel(self.temp_bert, True, True, attention_dropout, fc_dropout)
        self.config = self.bert.config
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)
        self.discriminator = nn.Linear(self.config.hidden_size, num_polit)
        del(self.temp_bert)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, task=True, grad = True):
        #with torch.set_grad_enabled(True):
        #outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        if task:
            outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, mode= 'politic')
            #sequence_output = outputs
            #pooled_output = self.pooler(sequence_output)
            pooled_output = self.dropout(outputs) 
            logits = self.classifier(pooled_output)
        else:
            #if grad:
            #self.generator.load_state_dict(self.discriminator.state_dict())
            #logits = self.generator(pooled_output)
            #else:
            if grad == True:
                outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, mode= 'metaphor')
                pooled_output = self.dropout(outputs) 
                logits = self.discriminator(pooled_output)
            else:
                outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, mode= 'politic')
                pooled_output = self.dropout(outputs) 
                logits = self.discriminator(pooled_output)           
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

class AdaptedBertModel(nn.Module):
    """
    BERT model ("Bidirectional Embedding Representations from a Transformer").

    Taken from the pytorch_pretrained_bert library.
    """
    def __init__(self, model, metaphor, politic, attention_dropout, fc_dropout):
        super().__init__()
        self.embeddings = model.embeddings
        self.encoder = BertEncoder(model.encoder.layer, metaphor, politic, attention_dropout, fc_dropout)
        self.config = model.config
        self.pooler = model.pooler
    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                mode="politic"):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        ) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        embeddings = self.embeddings(input_ids, token_type_ids)
        embeddings = self.encoder(embeddings, extended_attention_mask, mode)
        if mode == 'politic':
            embeddings = self.pooler(embeddings)
        return embeddings

class BertEncoder(nn.Module):
    def __init__(self, layers, metaphor, politic, attention_dropout, fc_dropout):
        super().__init__()
        self.layers = layers[:-1]
        self.output_attentions = False
        for layer in self.layers:
            layer.attention.self.dropout = nn.Dropout(attention_dropout)
        if metaphor:
            self.layer_left = copy.deepcopy(layers[-1])
        if politic:
            self.layer_right = copy.deepcopy(layers[-1])

    def forward(self, hidden, attention_mask, mode):
        all_attentions = ()
        #device = next(self.layers.parameters()).device
        #hidden = hidden.to(device)
        #attention_mask = attention_mask.to(device)
        #with torch.set_grad_enabled(False):
        for layer in self.layers:
            hidden = layer(hidden, attention_mask)
            if self.output_attentions:
                all_attentions = all_attentions + (hidden[1],)
            hidden = hidden[0]
        if mode == "metaphor":
            hidden = self.layer_left(hidden, attention_mask)
        elif mode == "politic":
            hidden = self.layer_right(hidden, attention_mask)
        outputs = hidden[0]# + (hidden[1],)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs

class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class DocumentBertAtt(nn.Module):
    """
    Roberta output over document in Attention
    """

    def __init__(self, language, cache_dir, num_labels, num_polit, bert_batch_size, attention_dropout, fc_dropout):
        super(DocumentBertAtt, self).__init__()

        '''self.bert = BertModel.from_pretrained(
                language, cache_dir=cache_dir, num_labels=num_labels
            )'''
        '''self.temp_bert = BertModel.from_pretrained(
                'bert-base-uncased', cache_dir=cache_dir, num_labels=num_labels
            )'''
        self.bert = AdaptedBertModel(RobertaModel.from_pretrained(
                'roberta-base', cache_dir=cache_dir, num_labels=num_labels
            ), True, True, attention_dropout, fc_dropout)
        self.config = self.bert.config
        self.bert_batch_size = bert_batch_size
        self.classes = num_labels
        self.dropout = nn.Dropout(p=fc_dropout)
        self.classifier = nn.Sequential(
            nn.Dropout(p=fc_dropout),
            nn.Linear(self.config.hidden_size, num_labels),
        )
        self.discriminator = nn.Sequential(
            nn.Dropout(p=fc_dropout),
            nn.Linear(self.config.hidden_size, num_polit),
        )
        #self.self_attention = BertSelfAttention(self.config)

        self.attention = AttentionModule(self.config.hidden_size,
            batch_first=True,
            layers=1,
            dropout=.0,
            non_linearity="tanh")
        self.batchnorm = nn.BatchNorm1d(self.config.hidden_size)

    #input_ids, token_type_ids, attention_masks
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, task=True, grad=True):
        #grad = False
        if task:
            bert_output = torch.zeros(size=(input_ids.shape[0],
                                                min(input_ids.shape[1],self.bert_batch_size),
                                                self.bert.config.hidden_size), dtype=torch.float, device='cuda')
            with torch.set_grad_enabled(True):
                for doc_id in range(input_ids.shape[0]):
                    # Pass through encoder:
                    bert_output[doc_id][:self.bert_batch_size] = self.dropout(self.bert(input_ids[doc_id][:self.bert_batch_size],
                                                    attention_mask=attention_mask[doc_id][:self.bert_batch_size], mode= 'politic'))
            #bert_output = self.self_attention(bert_output)[0]
            attention_output, _, _ = self.attention.forward(inputs = bert_output, attention_weights = attention_mask.float().mean(dim=2))
            del(bert_output)
            prediction = self.classifier(attention_output)
            prediction = prediction.view(-1, self.classes)
        else:
            if grad == True:
                outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, mode= 'metaphor')
                prediction = self.discriminator(outputs)
            else:
                outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, mode= 'politic')
                prediction = self.discriminator(outputs)
        return prediction

class AttentionModule(nn.Module):
    def __init__(
        self,
        attention_size,
        batch_first=True,
        layers=1,
        dropout=.0,
        non_linearity="tanh"):
        super(AttentionModule, self).__init__()

        self.batch_first = batch_first

        if non_linearity == "relu":
            activation = nn.ReLU()
        else:
            activation = nn.Tanh()

        modules = []
        for _ in range(layers - 1):
            modules.append(nn.Linear(attention_size, attention_size))
            modules.append(activation)
            modules.append(nn.Dropout(dropout))
        self.linear_layer = nn.Sequential(*modules)
        modules_att = []
        # last attention layer must output 1
        modules_att.append(nn.Linear(attention_size, 1))
        modules_att.append(activation)
        modules_att.append(nn.Dropout(dropout))

        self.attention = nn.Sequential(*modules_att)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs, attention_weights):

        ##################################################################
        # STEP 1 - perform dot product
        # of the attention vector and each hidden state
        ##################################################################

        # inputs is a 3D Tensor: batch, len, hidden_size
        # scores is a 2D Tensor: batch, len
        linear_output = self.linear_layer(inputs)
        scores = self.attention(linear_output).squeeze()
        scores = self.softmax(scores)

        ##################################################################
        # Step 3 - Weighted sum of hidden states, by the attention scores
        ##################################################################

        # multiply each hidden state with the attention weights
        weighted = torch.mul(linear_output, torch.mul(scores,attention_weights).unsqueeze(-1).expand_as(linear_output))

        # sum the hidden states
        representations = weighted.sum(1).squeeze()

        return representations, scores, weighted

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.output_attentions = config.output_attentions

        self.num_attention_heads = 6
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs