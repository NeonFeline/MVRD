  __  ____      _______  _____  
 |  \/  \ \    / /  __ \|  __ \ 
 | \  / |\ \  / /| |__) | |  | |
 | |\/| | \ \/ / |  _  /| |  | |
 | |  | |  \  /  | | \ \| |__| |
 |_|  |_|   \/   |_|  \_\_____/ 
                                
                                
### MVRD
## Maciek's Very Romantic Dream
# Transformer based chess engine with auxiliary thinking tokens

Provided here is the script for MVRD
Details:
- Transformer Encoder
- 24 layers
- 81 tokens (with 64 board tokens, 1 output token, 8 auxiliary thinking tokens)
- hidden size of 512
- SwiGLU size of 2024
- RMSNorm normalization
- Policy head
- Value Head

Trained on lichess Stockfish Evaluation dataset



"We can only see a short distance ahead, but we can see plenty there that needs to be done." - Alan Turing
