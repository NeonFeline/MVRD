import torch
import numpy as np
import chess
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

from model import ChessTransformer
from dataset.chess_dataset import ChessMoveTokenizer

print("Imports successful!")
