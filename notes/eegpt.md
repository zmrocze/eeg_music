

encoder + target_encoder - chodzi o to że używają dwóch kopii jako encoder i momentum encoder, przy czym momentum encoder jest updetowany jakoś z użyciem momentu (w opóźniony sposób??)

więc się tym nie stresować.
class LitEEGPT(pl.LightningModule): z pretraining to troche inne niż eegptclassifier z finetuning

no tak
target_encoder jest opóźniony względem encodera, updateowany zgodnie z w' = w*m + w_enc(1-m)
target_encoder enkoduje nie zamaskowane

trained on window_length 4*256

finetuning z tym eegpt z pracy:
 - na datasetach wielkości typu naszej (https://www.perplexity.ai/search/eegpt-pretrained-transformer-f-QjrJvJoXRqyCGtv94qWurQ) lub nawet dużo większej
 - i klasyfikacja na np 4 klasy. czyli warstwa liniowa na końcu jest ~32 razy mniejsza jeśli by wziąc 128 mel channels

z drugiej strony bi-lstm z neural decoding of music może mieć i ~4M a co najmniej 1M.

spróbujmy najpierw najprostsze: linear(512, 128),
hmmm albo może fc z dwóch sąsiednich patchy? żeby jednak nie zakładać że neural response jest w tym samym patchu
https://www.perplexity.ai/search/when-participant-is-listening-RWPqlSZcRMakIdl8M1QIpQ
wygląda że patch + jeden do przodu warto uwzględnić

kolejność kanałow chyba dowolna o ile podzbiór kanałów z treningu.

zróbmy:
uwzględniamy wszystkie z zarówno pretrain jak i calibration, ale również używamy chan_conv. i powinno być git
wtedy 28 kanałów

- [EEG_large](https://figshare.com/s/e37df4f8a907a866df4b) trained on mixed dataset (58-channels, 256Hz, 4s time length EEG) using patch size 64.

For downstream tasks, you should place it into `checkpoint` folder as file name "checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt". To use the model, simply load the checkpoint and pass it to the `EEGPTClassifier` class in "downstream/Modules/models/EEGPT_mcae_finetune.py".



# channels

prepare_chan_ids prepares index vector of ids
based on use_channel_names, (which have to be in CHANNEL_DICT)

then channel_conv changes dim from in_channels to img[0]

then 
patch_module(
  img_size=img_size,
  patch_size=patch_size,
  patch_stride=patch_stride,
  embed_dim=embed_dim
)

the channel_ids got from using_channels are used to grab matching channel_embeddings, so arbitrary order!

use_channels_names has to be 58?

solution:
so best to pick the channels and count for input (28)
pick all we have (32), and some arbitrary additional channel_dict to fill up to the number

solution 2:
just use the intersection of ours and finetuning ones.
then you don't even need chan_conv.
Q: so to use it or not?

A: when using it we need to backpropagate through whole network (even if we don't update transformer weights)
so let's set use_chan_conv to False!