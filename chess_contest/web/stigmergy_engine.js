/**
 * Stigmergy-DPFE browser engine — mirrors the Python field eval + deep search.
 * Rules come from chess.js; evaluation / search / book are custom.
 */
(function (global) {
  "use strict";

  const CHANNELS = 10;
  const PIECE_ORDER = "pnbrqk";
  const MATERIAL = { p: 100, n: 320, b: 330, r: 500, q: 900, k: 0 };
  const CODE_WEIGHT = { "-1": 1, "0": 3, "1": 6 };
  const MATE = 100000;
  const TIME_UP = Symbol("TIME_UP");

  const YS = [];
  const XS = [];
  for (let r = 0; r < 8; r++) {
    YS[r] = [];
    XS[r] = [];
    for (let c = 0; c < 8; c++) {
      YS[r][c] = r;
      XS[r][c] = c;
    }
  }
  const HARMONIC = Array.from({ length: 8 }, (_, r) =>
    Array.from({ length: 8 }, (_, c) => {
      return (
        Math.sin((Math.PI * (c + 1)) / 9) * Math.sin((Math.PI * (r + 1)) / 9) +
        0.5 * Math.sin((2 * Math.PI * (c + 1)) / 9) * Math.cos((Math.PI * (r + 1)) / 9)
      );
    })
  );

  function emptyField() {
    return Array.from({ length: CHANNELS }, () =>
      Array.from({ length: 8 }, () => Array(8).fill(0))
    );
  }

  function copyField(f) {
    return f.map((ch) => ch.map((row) => row.slice()));
  }

    function defaultField() {
      const deposit = [
        [1.0, 0.05, 1.2, 0.4, 0.1, 0.05, 0.0, 0.3, 0.2, 0.15],
        [3.2, 0.15, 0.0, 0.2, 0.8, 1.1, 0.6, 0.1, 0.5, 0.9],
        [3.3, 0.15, 0.0, 0.15, 1.4, 0.9, 0.7, 0.2, 0.4, 1.1],
        [5.0, 0.2, 0.1, 1.5, 0.3, 0.7, 0.5, 0.4, 0.3, 0.6],
        [9.0, 0.35, 0.1, 0.8, 0.6, 1.0, 0.9, 0.3, 0.6, 0.8],
        [0.0, 2.5, 0.0, 0.1, 0.2, 0.05, 0.0, 0.0, 0.1, 0.4],
      ];
      const interaction = Array.from({ length: CHANNELS }, (_, i) =>
        Array.from({ length: CHANNELS }, (_, j) => (i === j ? -0.35 : 0))
      );
      interaction[0][1] = -0.55;
      interaction[1][0] = -0.55;
      interaction[3][7] = 0.25;
      interaction[7][3] = -0.2;
      interaction[4][4] = -0.15;
      interaction[5][5] = 0.1;
      return {
        deposit,
        decay: Array(CHANNELS).fill(0.55),
        mix: Array(CHANNELS).fill(0.45),
        interaction,
        selfEnergy: [0.02, -0.08, 0.05, 0.04, 0.03, 0.06, 0.02, -0.03, 0.04, 0.02],
        kingResonance: [-0.4, -1.2, -0.1, -0.25, -0.15, -0.2, -0.35, -0.05, -0.1, -0.15],
        materialAnchor: 1.0,
        tempoBonus: 10.0,
        passedPawnScale: 1.15,
        mobilityScale: 1.25,
        fieldHead: Array(24).fill(0),
        swarmScale: 1.0,
      };
    }

  function dequantizeLearned(q) {
    if (!q || !q.codes) return {};
    const scale = q.scale || 0;
    const out = {};
    for (const k in q.codes) out[k] = q.codes[k] * scale;
    return out;
  }

  function loadWeightsPayload(data) {
    if (!data || (data.formatVersion || 0) < 2) {
      throw new Error("Need Stigmergy formatVersion >= 2 weights JSON");
    }
    const field = data.field || defaultField();
    return {
      field,
      book: data.book || {},
      learnedMoves: dequantizeLearned(data.learnedMoves),
      diffusionSteps: data.diffusionSteps || 3,
      meta: data.trainingMeta || {},
      fingerprint: data.uniquenessFingerprint || {},
    };
  }

  function sqRC(sq) {
    // chess.js board() is [rank8..rank1][file a..h] already as row/col.
    return null;
  }

  function deposit(chessInst, fieldParams) {
    const board = chessInst.board();
    const fw = emptyField();
    const fb = emptyField();
    let material = 0;
    let wKing = null;
    let bKing = null;
    const wPawns = Array.from({ length: 8 }, () => []);
    const bPawns = Array.from({ length: 8 }, () => []);

    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        const cell = board[r][c];
        if (!cell) continue;
        const idx = PIECE_ORDER.indexOf(cell.type);
        const seed = fieldParams.deposit[idx];
        const target = cell.color === "w" ? fw : fb;
        for (let ch = 0; ch < CHANNELS; ch++) target[ch][r][c] += seed[ch];
        if (idx === 0) {
          const rDep = cell.color === "w" ? r : 7 - r;
          target[2][r][c] += 0.15 * (rDep / 7);
        }
        if (idx === 2) {
          target[4][r][c] += (r + c) % 2 === 0 ? 0.35 : -0.35;
        }
        target[9][r][c] += 0.2 * HARMONIC[r][c] * (1 + 0.1 * idx);
        const sign = cell.color === "w" ? 1 : -1;
        material += sign * MATERIAL[cell.type];
        if (idx === 5) {
          if (cell.color === "w") wKing = { r, c };
          else bKing = { r, c };
        }
        if (idx === 0) (cell.color === "w" ? wPawns : bPawns)[c].push(r);
      }
    }
    return { fw, fb, material, wKing, bKing, wPawns, bPawns };
  }

  function diffuse(field, decay, mix, steps) {
    let out = copyField(field);
    for (let s = 0; s < steps; s++) {
      const nxt = emptyField();
      for (let ch = 0; ch < CHANNELS; ch++) {
        const d = decay[ch];
        const m = mix[ch];
        for (let r = 0; r < 8; r++) {
          for (let c = 0; c < 8; c++) {
            const up = out[ch][Math.max(0, r - 1)][c];
            const down = out[ch][Math.min(7, r + 1)][c];
            const left = out[ch][r][Math.max(0, c - 1)];
            const right = out[ch][r][Math.min(7, c + 1)];
            const neigh = 0.25 * (up + down + left + right);
            let v = (1 - m) * out[ch][r][c] + m * neigh;
            v *= d;
            v *= 1 / (0.85 + 0.15 * Math.max(...decay));
            nxt[ch][r][c] = v;
          }
        }
      }
      out = nxt;
    }
    return out;
  }

  function passedPawnScore(wPawns, bPawns) {
    let score = 0;
    const bonus = [0, 120, 90, 60, 40, 25, 12, 0];
    for (let f = 0; f < 8; f++) {
      for (const r of wPawns[f]) {
        let passed = true;
        for (const nf of [f - 1, f, f + 1]) {
          if (nf < 0 || nf > 7) continue;
          for (const br of bPawns[nf]) if (br < r) passed = false;
        }
        if (passed) score += bonus[r];
        if (wPawns[f].length > 1) score -= 12;
        if (!((f > 0 && wPawns[f - 1].length) || (f < 7 && wPawns[f + 1].length))) score -= 15;
      }
      for (const r of bPawns[f]) {
        let passed = true;
        for (const nf of [f - 1, f, f + 1]) {
          if (nf < 0 || nf > 7) continue;
          for (const wr of wPawns[nf]) if (wr > r) passed = false;
        }
        if (passed) score -= bonus[7 - r];
        if (bPawns[f].length > 1) score += 12;
        if (!((f > 0 && bPawns[f - 1].length) || (f < 7 && bPawns[f + 1].length))) score += 15;
      }
    }
    return score;
  }

  function evaluate(chessInst, weights) {
    if (chessInst.in_checkmate()) return chessInst.turn() === "w" ? -MATE : MATE;
    if (chessInst.in_draw()) return 0;
    const p = weights.field;
    const dep = deposit(chessInst, p);
    const fw = diffuse(dep.fw, p.decay, p.mix, weights.diffusionSteps);
    const fb = diffuse(dep.fb, p.decay, p.mix, weights.diffusionSteps);

    let bilinear = 0;
    let selfW = 0;
    let selfB = 0;
    for (let c = 0; c < CHANNELS; c++) {
      for (let d = 0; d < CHANNELS; d++) {
        let gram = 0;
        for (let r = 0; r < 8; r++) {
          for (let f = 0; f < 8; f++) gram += fw[c][r][f] * fb[d][r][f];
        }
        bilinear += p.interaction[c][d] * gram;
      }
      for (let r = 0; r < 8; r++) {
        for (let f = 0; f < 8; f++) {
          selfW += p.selfEnergy[c] * fw[c][r][f] * fw[c][r][f];
          selfB += p.selfEnergy[c] * fb[c][r][f] * fb[c][r][f];
        }
      }
    }
    let kingTerm = 0;
    if (dep.wKing) {
      for (let c = 0; c < CHANNELS; c++) kingTerm += p.kingResonance[c] * fb[c][dep.wKing.r][dep.wKing.c];
    }
    if (dep.bKing) {
      for (let c = 0; c < CHANNELS; c++) kingTerm -= p.kingResonance[c] * fw[c][dep.bKing.r][dep.bKing.c];
    }
    const material = dep.material * p.materialAnchor;
    const passed = passedPawnScore(dep.wPawns, dep.bPawns) * p.passedPawnScale;
    let mobility = chessInst.moves().length * p.mobilityScale;
    if (chessInst.turn() === "b") mobility = -mobility;
    const tempo = chessInst.turn() === "w" ? p.tempoBonus : -p.tempoBonus;
    let swarm = 0;
    if (p.fieldHead && p.fieldHead.length) {
      // Lightweight swarm readout (channel mean diffs + material).
      const feats = new Array(24).fill(0);
      for (let c = 0; c < Math.min(CHANNELS, 10); c++) {
        let mw = 0, mb = 0;
        for (let r = 0; r < 8; r++) for (let f = 0; f < 8; f++) {
          mw += fw[c][r][f]; mb += fb[c][r][f];
        }
        feats[c] = (mw - mb) / 64;
      }
      feats[12] = dep.material / 1000;
      const n = Math.min(feats.length, p.fieldHead.length);
      for (let i = 0; i < n; i++) swarm += p.fieldHead[i] * feats[i];
      swarm *= p.swarmScale || 1;
    }
    return material + passed + mobility + tempo + 18 * bilinear + 4 * (selfW - selfB) + 55 * kingTerm + 12 * swarm;
  }

  function relativeEval(chessInst, weights) {
    const s = evaluate(chessInst, weights);
    return chessInst.turn() === "w" ? s : -s;
  }

  function getBookMove(chessInst, weights) {
    const hist = chessInst.history({ verbose: true });
    const key = hist.map((h) => h.from + h.to).join("");
    const entry = weights.book[key];
    if (!entry || !entry.length) return null;
    let best = entry[0];
    for (const e of entry) {
      if ((e.code || 0) > (best.code || 0)) best = e;
    }
    return best.m;
  }

  function createSearcher(weights) {
    let nodes = 0;
    let deadline = 0;
    let tt = new Map();
    let killers = {};
    let history = {};

    function checkTime() {
      if ((nodes & 1023) === 0 && performance.now() > deadline) throw TIME_UP;
    }

    function moveKey(m) {
      return m.from + m.to + (m.promotion || "");
    }

    function moveScore(chessInst, m, ply, ttMove) {
      let s = 0;
      if (ttMove && m.from === ttMove.from && m.to === ttMove.to && m.promotion === ttMove.promotion) s += 1e6;
      const isCap = m.flags.includes("c") || m.flags.includes("e");
      if (isCap) {
        const victim = m.captured || "p";
        s += 10000 + 10 * MATERIAL[victim] - MATERIAL[m.piece];
      }
      if (m.promotion) s += 8000 + 100 * (MATERIAL[m.promotion] || 0);
      const k = killers[ply];
      if (k) {
        if (k[0] && k[0].from === m.from && k[0].to === m.to) s += 9000;
        else if (k[1] && k[1].from === m.from && k[1].to === m.to) s += 8000;
      }
      const hk = m.from + m.to;
      if (history[hk]) s += Math.min(history[hk], 5000);
      const lk = m.piece + m.from + m.to;
      if (weights.learnedMoves[lk]) s += weights.learnedMoves[lk] * 400;
      return s;
    }

    function orderMoves(chessInst, moves, ply, ttMove) {
      for (const m of moves) m._s = moveScore(chessInst, m, ply, ttMove);
      moves.sort((a, b) => b._s - a._s);
    }

    function quiesce(chessInst, alpha, beta, ply) {
      nodes++;
      checkTime();
      if (chessInst.in_checkmate()) return -(MATE - ply);
      if (chessInst.in_draw()) return 0;
      const stand = relativeEval(chessInst, weights);
      if (stand >= beta) return beta;
      if (alpha < stand) alpha = stand;
      if (ply > 48) return alpha;
      const moves = chessInst
        .moves({ verbose: true })
        .filter((m) => m.flags.includes("c") || m.flags.includes("e") || m.promotion || m.san.includes("+"));
      orderMoves(chessInst, moves, ply, null);
      for (const m of moves.slice(0, 24)) {
        chessInst.move(m);
        let score;
        try {
          score = -quiesce(chessInst, -beta, -alpha, ply + 1);
        } finally {
          chessInst.undo();
        }
        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
      }
      return alpha;
    }

    function negamax(chessInst, depth, alpha, beta, ply, allowNull) {
      nodes++;
      checkTime();
      if (chessInst.in_checkmate()) return -(MATE - ply);
      if (chessInst.in_draw() || chessInst.in_threefold_repetition()) return 0;
      const fenKey = chessInst.fen();
      const ttHit = tt.get(fenKey);
      let ttMove = null;
      if (ttHit && ttHit.depth >= depth) {
        if (ttHit.flag === 0) return ttHit.score;
        if (ttHit.flag === 1) alpha = Math.max(alpha, ttHit.score);
        if (ttHit.flag === -1) beta = Math.min(beta, ttHit.score);
        if (alpha >= beta) return ttHit.score;
        ttMove = ttHit.move;
      } else if (ttHit) ttMove = ttHit.move;

      const inCheck = chessInst.in_check();
      if (depth <= 0) return quiesce(chessInst, alpha, beta, ply);

      if (allowNull && depth >= 3 && !inCheck) {
        // Null move via FEN side flip is awkward in chess.js — skip null move in JS for safety.
      }

      const moves = chessInst.moves({ verbose: true });
      orderMoves(chessInst, moves, ply, ttMove);
      if (!moves.length) return inCheck ? -(MATE - ply) : 0;

      const alphaOrig = alpha;
      let best = -Infinity;
      let bestMove = moves[0];
      for (let i = 0; i < moves.length; i++) {
        const m = moves[i];
        let reduction = 0;
        if (depth >= 3 && i >= 4 && !inCheck && !m.captured && !m.promotion) {
          reduction = 1 + Math.floor(i / 8);
        }
        chessInst.move(m);
        let score;
        try {
          score = -negamax(chessInst, depth - 1 - reduction, -beta, -alpha, ply + 1, true);
          if (reduction && score > alpha) {
            score = -negamax(chessInst, depth - 1, -beta, -alpha, ply + 1, true);
          }
        } finally {
          chessInst.undo();
        }
        if (score > best) {
          best = score;
          bestMove = m;
        }
        if (best > alpha) alpha = best;
        if (alpha >= beta) {
          if (!m.captured && !m.flags.includes("e")) {
            if (!killers[ply]) killers[ply] = [null, null];
            killers[ply][1] = killers[ply][0];
            killers[ply][0] = { from: m.from, to: m.to };
            const hk = m.from + m.to;
            history[hk] = (history[hk] || 0) + depth * depth;
          }
          break;
        }
      }
      let flag = 0;
      if (best <= alphaOrig) flag = -1;
      else if (best >= beta) flag = 1;
      tt.set(fenKey, { depth, score: best, flag, move: bestMove });
      if (tt.size > 80000) tt.clear();
      return best;
    }

    function rootSearch(chessInst, depth, alpha, beta, pvMove) {
      const moves = chessInst.moves({ verbose: true });
      orderMoves(chessInst, moves, 0, pvMove);
      let best = -Infinity;
      let bestMove = moves[0];
      for (const m of moves) {
        chessInst.move(m);
        let score;
        try {
          score = -negamax(chessInst, depth - 1, -beta, -alpha, 1, true);
        } finally {
          chessInst.undo();
        }
        if (score > best) {
          best = score;
          bestMove = m;
        }
        if (best > alpha) alpha = best;
      }
      return { move: bestMove, score: best };
    }

    function findBestMove(chessInst, timeLimitMs, maxDepth) {
      const bookUci = getBookMove(chessInst, weights);
      if (bookUci) {
        const from = bookUci.slice(0, 2);
        const to = bookUci.slice(2, 4);
        const promo = bookUci.length > 4 ? bookUci[4] : undefined;
        const legal = chessInst.moves({ verbose: true }).find((m) => m.from === from && m.to === to && (!promo || m.promotion === promo));
        if (legal) return { move: legal, score: 0, depth: 0, nodes: 0, book: true };
      }
      const rootMoves = chessInst.moves({ verbose: true });
      if (!rootMoves.length) return { move: null, score: 0, depth: 0, nodes: 0, book: false };
      if (rootMoves.length === 1) return { move: rootMoves[0], score: 0, depth: 1, nodes: 1, book: false };

      // Mate-in-one probe.
      for (const m of rootMoves) {
        chessInst.move(m);
        const mate = chessInst.in_checkmate();
        chessInst.undo();
        if (mate) return { move: m, score: MATE - 1, depth: 1, nodes: rootMoves.length, book: false };
      }

      nodes = 0;
      deadline = performance.now() + Math.max(80, timeLimitMs);
      tt = new Map();
      killers = {};
      history = {};
      let bestMove = rootMoves[0];
      let bestScore = 0;
      let depthReached = 0;
      try {
        for (let d = 1; d <= (maxDepth || 10); d++) {
          if (d === 1) {
            const res = rootSearch(chessInst, d, -Infinity, Infinity, bestMove);
            bestMove = res.move;
            bestScore = res.score;
            depthReached = 1;
            continue;
          }
          let window = 80;
          let alpha = bestScore - window;
          let beta = bestScore + window;
          for (let attempt = 0; attempt < 5; attempt++) {
            const res = rootSearch(chessInst, d, alpha, beta, bestMove);
            if (res.score <= alpha) {
              alpha -= window;
              window *= 2;
              continue;
            }
            if (res.score >= beta) {
              beta += window;
              window *= 2;
              continue;
            }
            bestMove = res.move;
            bestScore = res.score;
            depthReached = d;
            break;
          }
          if (Math.abs(bestScore) > MATE - 1000) break;
          if (performance.now() >= deadline) break;
        }
      } catch (e) {
        if (e !== TIME_UP) throw e;
      }
      return { move: bestMove, score: bestScore, depth: depthReached, nodes, book: false };
    }

    return { findBestMove, evaluate: (c) => evaluate(c, weights) };
  }

  global.StigmergyEngine = {
    defaultField,
    loadWeightsPayload,
    createSearcher,
    evaluate,
    CHANNELS,
  };
})(typeof window !== "undefined" ? window : globalThis);
