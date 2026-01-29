<script setup>
import { onMounted, ref, onBeforeUnmount } from "vue";
import { Chess } from "chess.js";

const board = ref(null);
const game = ref(new Chess()); // Initialize game immediately
const gameOver = ref(false); // Track game state with a separate ref
const myBoardRef = ref(null);
const whiteSquareGrey = "#eebefa";
const blackSquareGrey = "#cc5de8";

function removeGreySquares() {
  if (!window.$) return;
  window.$("#myBoard .square-55d63").css("background", "");
}

function greySquare(square) {
  if (!window.$) return;
  const $square = window.$("#myBoard .square-" + square);
  let background = whiteSquareGrey;
  if ($square.hasClass("black-3c85d")) {
    background = blackSquareGrey;
  }
  $square.css("background", background);
}

function onDragStart(source, piece) {
  // Do not pick up pieces if the game is over
  if (game.value.isGameOver()) return false;

  // Do not pick up opponent's pieces
  if (
    (game.value.turn() === "w" && piece.search(/^b/) !== -1) ||
    (game.value.turn() === "b" && piece.search(/^w/) !== -1)
  ) {
    return false;
  }
}

async function makeAIMove() {
  try {
    // Send current FEN to the backend API
    const response = await fetch("http://localhost:9400/move/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ fen: game.value.fen() }),
    });
    const data = await response.json();
    if (data.best_move) {
      // data.best_move is expected to be a FEN string representing the board after the AI's move.
      game.value.load(data.best_move);
      board.value.position(game.value.fen());
      gameOver.value = game.value.isGameOver();
    }
  } catch (error) {
    console.error("Error fetching AI move:", error);
  }
}

function onDrop(source, target) {
  // Allow dropping on the same square.
  if (source === target) {
    onSnapEnd();
    return "snapback";
  }

  removeGreySquares();

  // Try making the player's move.
  const move = game.value.move({
    from: source,
    to: target,
    promotion: "q", // Always promote to a queen for simplicity.
  });

  // If the move is illegal, snap the piece back.
  if (move === null) return "snapback";

  // Update game over status.
  gameOver.value = game.value.isGameOver();

  // If the game is not over and it's now the computer’s turn (assuming computer plays black).
  if (!game.value.isGameOver() && game.value.turn() === "b") {
    // Optionally add a delay to mimic thinking time.
    setTimeout(() => {
      makeAIMove();
    }, 500);
  }
}

function onMouseoverSquare(square, piece) {
  // Get list of possible moves for this square.
  const moves = game.value.moves({
    square: square,
    verbose: true,
  });
  if (moves.length === 0) return;
  greySquare(square);
  // Highlight the squares for all possible moves.
  for (let i = 0; i < moves.length; i++) {
    greySquare(moves[i].to);
  }
}

function onMouseoutSquare(square, piece) {
  removeGreySquares();
}

function onSnapEnd() {
  board.value.position(game.value.fen());
}

// Handle window resize.
const handleResize = () => {
  if (board.value && board.value.resize) {
    board.value.resize();
  }
};

onMounted(async () => {
  // Load jQuery (required by chessboardjs).
  await new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src =
      "https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js";
    script.onload = resolve;
    script.onerror = reject;
    document.head.appendChild(script);
  });

  // Load chessboard.js (CSS and JS).
  await new Promise((resolve, reject) => {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href =
      "https://cdnjs.cloudflare.com/ajax/libs/chessboard-js/1.0.0/chessboard-1.0.0.min.css";
    document.head.appendChild(link);

    const script = document.createElement("script");
    script.src =
      "https://cdnjs.cloudflare.com/ajax/libs/chessboard-js/1.0.0/chessboard-1.0.0.min.js";
    script.onload = resolve;
    script.onerror = reject;
    document.head.appendChild(script);
  });

  // Initialize game over status.
  gameOver.value = game.value.isGameOver();

  const config = {
    draggable: true,
    position: "start",
    onDragStart: onDragStart,
    onDrop: onDrop,
    onMouseoutSquare: onMouseoutSquare,
    onMouseoverSquare: onMouseoverSquare,
    onSnapEnd: onSnapEnd,
  };

  // Create the chessboard instance.
  board.value = window.Chessboard("myBoard", config);

  // Add a resize listener.
  window.addEventListener("resize", handleResize);

  document.querySelectorAll("div").forEach((div) => {
    const classes = div.classList;
    const hasSquare = [...classes].some((cls) => cls.startsWith("square-"));
    const hasBlack = [...classes].some((cls) => cls.startsWith("black-"));
    const hasWhite = [...classes].some((cls) => cls.startsWith("white-"));

    if (hasSquare && hasBlack) {
      div.classList.add("black");
    }
    if (hasSquare && hasWhite) {
      div.classList.add("white");
    }
  });
});

// Clean up event listeners on unmount.
onBeforeUnmount(() => {
  window.removeEventListener("resize", handleResize);
});
</script>

<template>
  <div>
    <nav class="main_nav">
      <h2 class="logo">MVRD v0.1</h2>
    </nav>
    <content class="main_content">
      <div class="status mt-4">
        <p v-if="gameOver" class="game_over">Game over</p>
      </div>
      <div class="game">
        <div id="myBoard" ref="myBoardRef" style="width: 400px;"></div>
      </div>
    </content>
  </div>
</template>

<style scoped>
.main_nav {
  /* box-shadow: 0 0 1rem 1rem rgba(0, 0, 0, 0.1); */
}

.game_over {
  font-size: 2.4rem;
  color: #5f3dc4;
  font-family: "Dosis", sans-serif;
}

.game {
  display: flex;
  justify-content: center;
  align-items: center;

  width: 100vw;
}

.main_content {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  height: 70vh;
}

.logo {
  padding: 2.4rem 4.8rem;
  font-family: "Dosis", sans-serif;
  color: #5f3dc4;
  font-size: 2.4rem;
}

#myBoard {
  max-width: 100%;
}
.status {
  margin-top: 10px;
  font-weight: bold;
}
</style>
