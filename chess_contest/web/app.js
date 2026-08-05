/* UI glue for the Stigmergy play site. */
(function () {
  "use strict";

  const chess = new Chess();
  const boardEl = document.getElementById("board");
  const statusEl = document.getElementById("status");
  const statsEl = document.getElementById("stats");
  const timeSelect = document.getElementById("timeSelect");
  const sideSelect = document.getElementById("sideSelect");
  const promoModal = document.getElementById("promoModal");
  const uniqBox = document.getElementById("uniqBox");
  const weightsStatus = document.getElementById("weightsStatus");

  let weights = null;
  let searcher = null;
  let squareEls = [];
  let selected = null;
  let legalTargets = [];
  let lastMove = null;
  let gameOver = false;
  let engineThinking = false;
  let humanColor = "w";

  const UNICODE = {
    w: { p: "♙", n: "♘", b: "♗", r: "♖", q: "♕", k: "♔" },
    b: { p: "♟", n: "♞", b: "♝", r: "♜", q: "♛", k: "♚" },
  };

  function log(msg) {
    const box = document.getElementById("trainLogBox");
    box.textContent += `[${new Date().toLocaleTimeString()}] ${msg}\n`;
    box.scrollTop = box.scrollHeight;
  }

  function squareName(r, c) {
    return "abcdefgh"[c] + (8 - r);
  }

  function buildBoardDOM() {
    boardEl.innerHTML = "";
    const flip = humanColor === "b";
    const grid = Array.from({ length: 8 }, () => new Array(8));
    for (let vr = 0; vr < 8; vr++) {
      for (let vc = 0; vc < 8; vc++) {
        const r = flip ? 7 - vr : vr;
        const c = flip ? 7 - vc : vc;
        const div = document.createElement("div");
        div.className = "sq " + ((r + c) % 2 === 0 ? "light" : "dark");
        div.addEventListener("click", () => onSquareClick(r, c));
        boardEl.appendChild(div);
        grid[r][c] = div;
      }
    }
    squareEls = grid;
  }

  function render() {
    if (!squareEls.length) return;
    const grid = chess.board();
    let checkSq = null;
    if (chess.in_check()) {
      const mover = chess.turn();
      for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
          const cell = grid[r][c];
          if (cell && cell.type === "k" && cell.color === mover) checkSq = squareName(r, c);
        }
      }
    }
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        const div = squareEls[r][c];
        const cell = grid[r][c];
        const sq = squareName(r, c);
        div.innerHTML = "";
        div.className = "sq " + ((r + c) % 2 === 0 ? "light" : "dark");
        if (cell) {
          const span = document.createElement("span");
          span.className = "piece " + (cell.color === "w" ? "wp" : "bp");
          span.textContent = UNICODE[cell.color][cell.type];
          div.appendChild(span);
        }
        if (lastMove && (sq === lastMove.from || sq === lastMove.to)) div.classList.add("last-move");
        if (sq === selected) div.classList.add("selected");
        if (legalTargets.includes(sq)) div.classList.add(cell ? "legal-capture" : "legal-dot");
        if (sq === checkSq) div.classList.add("in-check");
      }
    }
    boardEl.classList.toggle(
      "disabled",
      !weights || engineThinking || gameOver || chess.turn() !== humanColor
    );
  }

  function refreshWeightsUI() {
    if (!weights) {
      weightsStatus.textContent = "No weights loaded.";
      uniqBox.textContent = "—";
      document.getElementById("newGameBtn").disabled = true;
      document.getElementById("undoBtn").disabled = true;
      return;
    }
    const f = weights.field;
    weightsStatus.innerHTML = [
      `<b>engine</b> stigmergy-dpfe`,
      `channels: ${StigmergyEngine.CHANNELS} · diffusion: ${weights.diffusionSteps}`,
      `materialAnchor: ${Number(f.materialAnchor).toFixed(3)}`,
      `tempoBonus: ${Number(f.tempoBonus).toFixed(2)}`,
      `book positions: ${Object.keys(weights.book || {}).length}`,
      `learned move codes: ${Object.keys(weights.learnedMoves || {}).length}`,
    ].join("<br>");
    const fp = weights.fingerprint || {};
    uniqBox.textContent = `family: ${fp.family || "?"} · features: ${(fp.features || []).join(", ") || "—"}`;
    document.getElementById("newGameBtn").disabled = false;
    document.getElementById("undoBtn").disabled = false;
  }

  function applyWeights(payload) {
    weights = StigmergyEngine.loadWeightsPayload(payload);
    searcher = StigmergyEngine.createSearcher(weights);
    refreshWeightsUI();
    log("Weights loaded.");
  }

  async function loadBundledBase() {
    try {
      const resp = await fetch("../weights/base_weights.json");
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      applyWeights(await resp.json());
    } catch (err) {
      log("Bundled base not reachable (" + err.message + "). Use Import or serve from chess_contest/.");
      // Fall back to baked-in defaults so the site is playable offline.
      applyWeights({
        formatVersion: 2,
        engine: "stigmergy-dpfe",
        architecture: "diffusive-pheromone-field",
        channels: 10,
        diffusionSteps: 3,
        field: StigmergyEngine.defaultField(),
        book: {
          "": [
            { m: "e2e4", code: 1 },
            { m: "d2d4", code: 1 },
            { m: "c2c4", code: 0 },
            { m: "g1f3", code: 0 },
          ],
          e2e4: [
            { m: "e7e5", code: 1 },
            { m: "c7c5", code: 1 },
            { m: "e7e6", code: 0 },
          ],
          d2d4: [
            { m: "d7d5", code: 1 },
            { m: "g8f6", code: 1 },
          ],
        },
        learnedMoves: { codes: {}, scale: 0 },
        uniquenessFingerprint: {
          family: "diffusive-pheromone-field",
          features: [
            "multi_channel_pheromone_lattice",
            "jacobi_diffusion_kernels",
            "bilinear_cross_color_interaction",
            "king_resonance_coupling",
            "ternary_ant_trail_book",
            "harmonic_board_channel",
          ],
          not: ["stockfish", "nnue", "alphazero", "lc0", "classic-pst-only"],
        },
      });
    }
  }

  function onSquareClick(r, c) {
    if (!weights || gameOver || engineThinking || chess.turn() !== humanColor) return;
    const sq = squareName(r, c);
    const cell = chess.board()[r][c];
    if (selected) {
      if (sq === selected) {
        selected = null;
        legalTargets = [];
        render();
        return;
      }
      if (legalTargets.includes(sq)) {
        handlePlayerMove(selected, sq);
        return;
      }
      if (cell && cell.color === humanColor) {
        selected = sq;
        legalTargets = chess.moves({ square: sq, verbose: true }).map((m) => m.to);
        render();
        return;
      }
      selected = null;
      legalTargets = [];
      render();
      return;
    }
    if (cell && cell.color === humanColor) {
      selected = sq;
      legalTargets = chess.moves({ square: sq, verbose: true }).map((m) => m.to);
      render();
    }
  }

  function showPromotionModal() {
    promoModal.style.display = "flex";
    return new Promise((resolve) => {
      const handler = (e) => {
        const p = e.target.getAttribute("data-p");
        if (!p) return;
        promoModal.style.display = "none";
        promoModal.removeEventListener("click", handler);
        resolve(p);
      };
      promoModal.addEventListener("click", handler);
    });
  }

  function handlePlayerMove(from, to) {
    const candidates = chess.moves({ square: from, verbose: true }).filter((m) => m.to === to);
    if (!candidates.length) return;
    if (candidates[0].flags.includes("p")) {
      showPromotionModal().then((piece) => {
        chess.move({ from, to, promotion: piece });
        finishPlayerMove();
      });
    } else {
      chess.move(candidates[0]);
      finishPlayerMove();
    }
  }

  function finishPlayerMove() {
    const h = chess.history({ verbose: true });
    lastMove = h[h.length - 1];
    selected = null;
    legalTargets = [];
    render();
    if (checkGameOver()) return;
    engineThinking = true;
    render();
    statusEl.textContent = "Stigmergy is thinking (field search)…";
    setTimeout(engineMoveLive, 40);
  }

  function checkGameOver() {
    if (chess.in_checkmate()) {
      const winner = chess.turn() === "w" ? "b" : "w";
      statusEl.textContent =
        "Checkmate — " + (winner === humanColor ? "You" : "Stigmergy") + " wins.";
      gameOver = true;
      render();
      return true;
    }
    if (chess.in_stalemate()) {
      statusEl.textContent = "Draw — stalemate.";
      gameOver = true;
      render();
      return true;
    }
    if (chess.in_threefold_repetition()) {
      statusEl.textContent = "Draw — repetition.";
      gameOver = true;
      render();
      return true;
    }
    if (chess.insufficient_material()) {
      statusEl.textContent = "Draw — insufficient material.";
      gameOver = true;
      render();
      return true;
    }
    if (chess.in_draw()) {
      statusEl.textContent = "Draw.";
      gameOver = true;
      render();
      return true;
    }
    return false;
  }

  function engineMoveLive() {
    const t = parseInt(timeSelect.value, 10);
    const maxDepth = t >= 10000 ? 12 : t >= 4000 ? 10 : t >= 1500 ? 8 : 6;
    const res = searcher.findBestMove(chess, t, maxDepth);
    if (!res.move) {
      statusEl.textContent = "No move found.";
      engineThinking = false;
      render();
      return;
    }
    chess.move(res.move);
    const h = chess.history({ verbose: true });
    lastMove = h[h.length - 1];
    engineThinking = false;
    render();
    if (checkGameOver()) return;
    if (res.book) {
      statusEl.textContent = "Your move. (ternary ant-trail book)";
      statsEl.textContent = "";
    } else {
      const pov = ((humanColor === "w" ? 1 : -1) * -res.score) / 100;
      statusEl.textContent = "Your move.";
      statsEl.textContent =
        "Depth " +
        res.depth +
        " · Nodes " +
        res.nodes.toLocaleString() +
        " · Eval " +
        (pov >= 0 ? "+" : "") +
        pov.toFixed(2) +
        " (your POV)";
    }
  }

  function startNewGame() {
    if (!weights) return;
    chess.reset();
    selected = null;
    legalTargets = [];
    lastMove = null;
    gameOver = false;
    engineThinking = false;
    statsEl.textContent = "";
    buildBoardDOM();
    render();
    if (humanColor === "w") {
      statusEl.textContent = "Your move (White).";
    } else {
      statusEl.textContent = "Stigmergy (White) is thinking…";
      engineThinking = true;
      render();
      setTimeout(engineMoveLive, 40);
    }
  }

  document.getElementById("newGameBtn").addEventListener("click", startNewGame);
  sideSelect.addEventListener("change", (e) => {
    humanColor = e.target.value;
    if (weights) startNewGame();
  });
  document.getElementById("undoBtn").addEventListener("click", () => {
    if (engineThinking || !weights) return;
    if (!chess.history().length) return;
    chess.undo();
    if (chess.history().length && chess.turn() !== humanColor) chess.undo();
    gameOver = false;
    selected = null;
    legalTargets = [];
    const h = chess.history({ verbose: true });
    lastMove = h.length ? h[h.length - 1] : null;
    statusEl.textContent = "Move undone.";
    statsEl.textContent = "";
    render();
  });
  document.getElementById("importFile").addEventListener("change", (e) => {
    const f = e.target.files[0];
    if (!f) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        applyWeights(JSON.parse(reader.result));
        startNewGame();
      } catch (err) {
        log("Import failed: " + err.message);
      }
    };
    reader.readAsText(f);
  });
  document.getElementById("loadBaseBtn").addEventListener("click", () => loadBundledBase().then(() => startNewGame()));
  document.getElementById("resetBtn").addEventListener("click", () => {
    weights = null;
    searcher = null;
    refreshWeightsUI();
    log("Cleared weights.");
  });

  buildBoardDOM();
  render();
  refreshWeightsUI();
  loadBundledBase().then(() => {
    log("Ready. Load trained weights for peak strength; bundled/default plays already.");
  });
})();
