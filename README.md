<pre>
__  ____      _______  _____  
 |  \/  \ \    / /  __ \|  __ \ 
 | \  / |\ \  / /| |__) | |  | |
 | |\/| | \ \/ / |  _  /| |  | |
 | |  | |  \  /  | | \ \| |__| |
 |_|  |_|   \/   |_|  \_\_____/ 
</pre>

### MVRD: Maciek's Very Romantic Dream
"We can only see a short distance ahead, but we can see plenty there that needs to be done." 
— Alan Turing

[PROJECT SCOPE]
A Transformer-based chess engine designed to explore the intersection of 
generative architecture and strategic intuition. 

[ARCHITECTURAL SPECIFICATIONS]
- Type:         Transformer Encoder (Pure Attention)
- Depth:        24 Layers
- Context:      81 Tokens 
                (64 Board | 8 Auxiliary Thinking | 1 Output/Action)
- Latent Dim:   512 (Hidden Size)
- MLP/Foresight: 2024 (SwiGLU)
- Normalization: RMSNorm
- Dual Head:    Policy (Move Prediction) & Value (Position Evaluation)

[TRAINING DATA]
- Dataset:      Lichess Stockfish Evaluation (Distillation Learning)
