# CPU generations — checkpoint 778.24M ids

Date: 2026-10-13. Checkpoint: `checkpoints/atom_latest/fractus_atom_1b_gpu0.bin` (`ids_processed` = 778,240,000).
Box: 8 vCPU, 15 GB RAM (shared with other jobs), fp32 weights memory-mapped (`torch.load(mmap=True)` + `load_state_dict(assign=True)`), `torch.inference_mode`, all threads.
Decoding: `fractus.generate_aligned.generate_window(..., carry=True, window=4096)`, prompt via `AtomFractusTokenizer.encode_open` (BOS, no EOS), carry state starts from zero. top_k=40, 64 new ids. Speed ~0.8-2.0 ids/s (machine loaded by other jobs); peak RSS 4.4 GB.

Teacher-forced CE on held-out corpus text, CPU fp32, same `tick_chunk` forward, SEQ=128 chunks with carry: ~6.3 nats/id from a zero state (first chunk 17-24), ~3.2-3.6 from a carried training state. The trainer's `ema_tf` (~1.1) is measured with a warm per-lane state; see `docs/LOSS_VS_GEN.md` (cold vs warm TF).

Earlier sample, sliding-window decode without carry (`carry=False`, window 128, bf16), temperature 0: prompt `What is the capital of France?` → `Hoaide acipet bedertmntion mm ataoderetableattmere coreti` (0.39 ids/s).

## 'What is the capital of France?' — temperature 0.0, top_k 40

```
aleeeeeeeeeeeedddeeeeeeeeeeerrrrrrrrrree                        
```
ids: `[97, 108, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 100, 100, 100, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 101, 101, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32]`  

## 'What is the capital of France?' — temperature 0.7, top_k 40

```
leeeeeeeeeeedddddeeeeeeeeeeeerrreeeeeeeee                      
```
ids: `[258, 108, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 100, 100, 100, 100, 100, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 114, 114, 114, 101, 101, 101, 101, 101, 101, 101, 101, 101, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32]`  

## 'What is the capital of France?' — temperature 1.0, top_k 40

```
alleeeeeeeeeeedddeeeeeeeeeeerrrrrrrrrrre                        
```
ids: `[97, 108, 108, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 100, 100, 100, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 101, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32]`  

## 'Bonjour, comment ça va ?' — temperature 0.0, top_k 40

```
udddddddddddllllllliiiiiiiiiiiiiiiiiiiiiiiiiiiiiiccccccccccccccc
```
ids: `[117, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 108, 108, 108, 108, 108, 108, 108, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99]`  

## 'Bonjour, comment ça va ?' — temperature 0.7, top_k 40

```
udddddddddddllllllliiiiiiiiiiiiiiiiiiiiiiiiiiiiiiccccccccccccccc
```
ids: `[117, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 108, 108, 108, 108, 108, 108, 108, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99]`  

## 'Bonjour, comment ça va ?' — temperature 1.0, top_k 40

```
udddddddddddllllllliiiiiiiiiiiiiiiiiiiiiiiiiiiiiiccccccccccccccc
```
ids: `[117, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 108, 108, 108, 108, 108, 108, 108, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99]`  

## 'Write a short poem about the sea.' — temperature 0.0, top_k 40

```
ceeeeeeeeeaaaaaaaaaaaaaaaaaaaaaaa                               
```
ids: `[99, 101, 101, 101, 101, 101, 101, 101, 101, 101, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32]`  

## 'Write a short poem about the sea.' — temperature 0.7, top_k 40

```
ceeeeeeeeeaaaaaaaaaaaaaaaaaaaaaaa                               
```
ids: `[99, 101, 101, 101, 101, 101, 101, 101, 101, 101, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32]`  

## 'Write a short poem about the sea.' — temperature 1.0, top_k 40

```
ceeeeeeeeeaaaaaaaaaaaaaaaaaaaaaaa                               
```
ids: `[99, 101, 101, 101, 101, 101, 101, 101, 101, 101, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32]`  

## 'Explique-moi la photosynthèse en deux phrases.' — temperature 0.0, top_k 40

```
oooaaaaaaaaaaarrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrraaaaaaaaaaaa
```
ids: `[111, 111, 111, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97]`  

## 'Explique-moi la photosynthèse en deux phrases.' — temperature 0.7, top_k 40

```
oooaaaaaaaaaaarrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrraaaaaaaaaaaa
```
ids: `[111, 111, 111, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97]`  

## 'Explique-moi la photosynthèse en deux phrases.' — temperature 1.0, top_k 40

```
oooaaaaaaaaaaarrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrraaaaaaaaaaaa
```
ids: `[111, 111, 111, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97]`  

## "Let's think step by step. If a train travels 60 km in 1 hour, then" — temperature 0.0, top_k 40

```
dddddddddddddddddddddddddddddddddaaaaaaaaaaaaaaaaaaaaaaaaaaaaaxx
```
ids: `[100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 120, 120]`  

## "Let's think step by step. If a train travels 60 km in 1 hour, then" — temperature 0.7, top_k 40

```
cddddddddddddddddddddddddddddddddaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
```
ids: `[99, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97]`  

## "Let's think step by step. If a train travels 60 km in 1 hour, then" — temperature 1.0, top_k 40

```
dddddddddddddddddddddddddddddddddaaaaaaaaaaaaaaaaaaaaaaaaaaaaaxx
```
ids: `[100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 120, 120]`  

## 'Il était une fois' — temperature 0.0, top_k 40

```
aaaaaaaddddddddddddddddddddddddddddddddddddddddddddddddddddddddd
```
ids: `[97, 97, 97, 97, 97, 97, 97, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]`  

## 'Il était une fois' — temperature 0.7, top_k 40

```
aaaaaaaddddddddddddddddddddddddddddddddddddddddddddddddddddddddd
```
ids: `[97, 97, 97, 97, 97, 97, 97, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]`  

## 'Il était une fois' — temperature 1.0, top_k 40

```
aaaaaaaadddddddddddddddddddddddddddddddddddddddddddddddddddddddd
```
ids: `[97, 97, 97, 97, 97, 97, 97, 97, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]`  
