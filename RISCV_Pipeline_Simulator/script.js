/* ============================================================
   RISC-V 5-STAGE PIPELINE SIMULATOR — ENGINE
   Stages: IF -> ID -> EX -> MEM -> WB
   Detects: Data Hazards (RAW) with EX/MEM + MEM/WB forwarding,
            Load-Use hazards (1 stall),
            Control Hazards on branches (1 cycle flush)
   ============================================================ */

const STAGES = ["IF", "ID", "EX", "MEM", "WB"];

// ---------- RISC-V Register File (x0-x31, ABI names) ----------
const ABI_NAMES = ["zero","ra","sp","gp","tp","t0","t1","t2","s0","s1",
  "a0","a1","a2","a3","a4","a5","a6","a7",
  "s2","s3","s4","s5","s6","s7","s8","s9","s10","s11",
  "t3","t4","t5","t6"];

let registers = {};
function regKey(i){ return "x" + i; }
function initRegisters(){
  for (let i = 0; i < 32; i++) registers[regKey(i)] = (i === 0) ? 0 : ((i % 7) + 1) * 2;
}

// ---------- Simulation state ----------
let program = [];
let schedule = [];
let totalCycles = 0;
let stallCount = 0;
let hazardEvents = [];
let currentCycle = 0;
let maxCycle = 0;
let playTimer = null;

// RISC-V instruction categories
const R_TYPE = ["add","sub","and","or","xor","sll","srl","sra","slt","mul","div"];
const I_TYPE_ARITH = ["addi","andi","ori","xori","slti"];
const LOAD_OPS = ["lw","lh","lb","ld"];
const STORE_OPS = ["sw","sh","sb","sd"];
const BRANCH_OPS = ["beq","bne","blt","bge"];

// ===================================================
// PARSER  (RISC-V assembly syntax)
// ===================================================
function normReg(tok) {
  if (!tok) return null;
  tok = tok.trim();
  const abiIdx = ABI_NAMES.indexOf(tok);
  if (abiIdx >= 0) return "x" + abiIdx;
  if (/^x([0-9]|[12][0-9]|3[01])$/.test(tok)) return tok;
  return null;
}

function parseProgram(text) {
  const rawLines = text.split("\n").map(l => l.trim()).filter(l => l.length > 0);
  const instructions = [];

  rawLines.forEach((line) => {
    line = line.split("#")[0].split("//")[0].trim();
    if (!line) return;
    // strip label "loop:" prefix if present, keep separately
    let label = null;
    const labelMatch = line.match(/^([A-Za-z_][\w]*):\s*(.*)$/);
    if (labelMatch) { label = labelMatch[1]; line = labelMatch[2].trim(); }
    if (!line) return;

    const opMatch = line.match(/^([A-Za-z][\w.]*)\s*(.*)$/);
    if (!opMatch) return;
    const op = opMatch[1].toLowerCase();
    const rest = opMatch[2].trim();

    let instr = { raw: line, op, label, dest: null, src1: null, src2: null,
      isLoad: false, isStore: false, isBranch: false };

    if (R_TYPE.includes(op)) {
      const parts = rest.split(",").map(p => p.trim());
      instr.dest = normReg(parts[0]);
      instr.src1 = normReg(parts[1]);
      instr.src2 = normReg(parts[2]);
    } else if (I_TYPE_ARITH.includes(op)) {
      const parts = rest.split(",").map(p => p.trim());
      instr.dest = normReg(parts[0]);
      instr.src1 = normReg(parts[1]);
      instr.imm = parts[2];
    } else if (LOAD_OPS.includes(op)) {
      // lw rd, offset(rs1)
      const parts = rest.split(",").map(p => p.trim());
      instr.dest = normReg(parts[0]);
      const m = parts[1] ? parts[1].match(/(-?\d+)?\(([^)]+)\)/) : null;
      instr.offset = m ? (m[1] || "0") : "0";
      instr.src1 = m ? normReg(m[2]) : null;
      instr.isLoad = true;
    } else if (STORE_OPS.includes(op)) {
      // sw rs2, offset(rs1)
      const parts = rest.split(",").map(p => p.trim());
      instr.src2 = normReg(parts[0]); // value being stored
      const m = parts[1] ? parts[1].match(/(-?\d+)?\(([^)]+)\)/) : null;
      instr.offset = m ? (m[1] || "0") : "0";
      instr.src1 = m ? normReg(m[2]) : null; // base address reg
      instr.isStore = true;
    } else if (BRANCH_OPS.includes(op)) {
      const parts = rest.split(",").map(p => p.trim());
      instr.src1 = normReg(parts[0]);
      instr.src2 = normReg(parts[1]);
      instr.target = parts[2];
      instr.isBranch = true;
    } else if (op === "jal") {
      const parts = rest.split(",").map(p => p.trim());
      instr.dest = normReg(parts[0]) || "x1";
      instr.target = parts[1] || parts[0];
      instr.isBranch = true;
    } else if (op === "nop") {
      // no operands
    }

    instructions.push(instr);
  });

  return instructions;
}

// ===================================================
// SCHEDULER — cycle-accurate model with RAW hazard
// detection (EX/MEM + MEM/WB forwarding), load-use
// stalls, and control hazard (1-cycle) flush.
// ===================================================
function buildSchedule(instructions) {
  const sched = instructions.map(() => ({ extraStall: 0 }));
  hazardEvents = [];

  let ifCycleArr = [];
  let idCycle = [];
  let nextIF = 1;

  for (let i = 0; i < instructions.length; i++) {
    ifCycleArr[i] = Math.max(nextIF, (i > 0 ? ifCycleArr[i - 1] + 1 : 1));
    let tentativeID = ifCycleArr[i] + 1;

    let stallsNeeded = 0;
    const instr = instructions[i];
    const usesRegs = [instr.src1, instr.src2].filter(r => r && r !== "x0");

    if (usesRegs.length > 0) {
      // check the two instructions immediately ahead (still in flight)
      for (let back = 1; back <= 2; back++) {
        const j = i - back;
        if (j < 0) continue;
        const prevInstr = instructions[j];
        if (!prevInstr.dest || prevInstr.dest === "x0") continue;
        if (!usesRegs.includes(prevInstr.dest)) continue;

        const prevEX = ifCycleArr[j] + 2 + (sched[j].extraStall || 0);
        let depEX = tentativeID + 1 + stallsNeeded;

        if (prevInstr.isLoad) {
          const prevMEM = prevEX + 1;
          if (depEX <= prevMEM) {
            const needed = (prevMEM + 1) - depEX;
            stallsNeeded += needed;
            hazardEvents.push({
              cycle: tentativeID + stallsNeeded,
              type: "data",
              message: `Load-Use Hazard: "${instr.raw}" needs ${prevInstr.dest} from "${prevInstr.raw}" before it's loaded → ${needed} stall cycle(s), then forwarded from MEM/WB.`
            });
          }
        } else {
          const needed = Math.max(0, (prevEX + 1) - depEX);
          if (needed > 0) {
            stallsNeeded += needed;
            hazardEvents.push({
              cycle: tentativeID + stallsNeeded,
              type: "data",
              message: `Data Hazard (RAW): "${instr.raw}" depends on ${prevInstr.dest} from "${prevInstr.raw}" → ${needed} stall cycle(s) inserted, then forwarded.`
            });
          } else {
            hazardEvents.push({
              cycle: tentativeID,
              type: "forward",
              message: `Data Hazard (RAW): "${instr.raw}" depends on ${prevInstr.dest} from "${prevInstr.raw}" → resolved via EX/MEM forwarding (no stall needed).`
            });
          }
        }
      }
    }

    sched[i].extraStall = stallsNeeded;
    idCycle[i] = tentativeID + stallsNeeded;
    nextIF = stallsNeeded > 0 ? ifCycleArr[i] + 1 + stallsNeeded : ifCycleArr[i] + 1;

    if (instr.isBranch) {
      hazardEvents.push({
        cycle: idCycle[i] + 1,
        type: "control",
        message: `Control Hazard: branch "${instr.raw}" resolved in EX → 1 cycle flush applied (predict not-taken).`
      });
      nextIF += 1;
    }
  }

  for (let i = 0; i < instructions.length; i++) {
    const IF = ifCycleArr[i];
    const ID = idCycle[i];
    const EX = ID + 1;
    const MEM = EX + 1;
    const WB = MEM + 1;
    sched[i] = { IF, ID, EX, MEM, WB, stalls: sched[i].extraStall || 0 };
  }

  stallCount = sched.reduce((sum, s) => sum + s.stalls, 0) +
               instructions.filter(ins => ins.isBranch).length;

  const lastWB = instructions.length ? Math.max(...sched.map(s => s.WB)) : 0;
  return { sched, totalCycles: lastWB };
}

// ===================================================
// RENDERING
// ===================================================
function buildPipelineTable(instructions, sched, totalCycles) {
  const head = document.getElementById("pipelineHead");
  const body = document.getElementById("pipelineBody");
  head.innerHTML = "";
  body.innerHTML = "";

  const thInstr = document.createElement("th");
  thInstr.textContent = "Instruction";
  thInstr.className = "instr-head";
  head.appendChild(thInstr);

  for (let c = 1; c <= totalCycles; c++) {
    const th = document.createElement("th");
    th.textContent = "C" + c;
    head.appendChild(th);
  }

  instructions.forEach((instr, i) => {
    const tr = document.createElement("tr");
    const tdLabel = document.createElement("td");
    tdLabel.className = "instr-label";
    tdLabel.innerHTML = `<span class="idx">I${i+1}</span> ${escapeHtml(instr.raw)}`;
    tr.appendChild(tdLabel);

    const s = sched[i];
    const stageOfCycle = {};
    STAGES.forEach(st => { stageOfCycle[s[st]] = st; });

    for (let c = 1; c <= totalCycles; c++) {
      const td = document.createElement("td");
      td.id = `cell-${i}-${c}`;
      const stage = stageOfCycle[c];
      if (stage) {
        const div = document.createElement("div");
        div.className = `stage-cell ${stage}`;
        div.textContent = stage;
        div.dataset.instr = i;
        div.dataset.stage = stage;
        td.appendChild(div);
      } else if (s.stalls > 0 && c > s.IF && c < s.EX && c !== s.ID) {
        const div = document.createElement("div");
        div.className = "stage-cell bubble";
        div.textContent = "stall";
        td.appendChild(div);
      }
      tr.appendChild(td);
    }
    body.appendChild(tr);
  });
}

function escapeHtml(str){
  return str.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

function revealUpToCycle(cycle) {
  document.querySelectorAll(".stage-cell").forEach(el => el.classList.remove("show", "current"));
  for (let c = 1; c <= cycle; c++) {
    document.querySelectorAll(`td[id$="-${c}"] .stage-cell`).forEach(el => {
      el.classList.add("show");
      if (c === cycle) el.classList.add("current");
    });
  }
  // progress bar
  const bar = document.getElementById("progressFill");
  if (bar && maxCycle > 0) bar.style.width = (Math.min(cycle, maxCycle) / maxCycle * 100) + "%";
}

function renderHazardLog(upToCycle) {
  const log = document.getElementById("hazardLog");
  const visible = hazardEvents.filter(e => e.cycle <= upToCycle);
  if (visible.length === 0) {
    log.innerHTML = `<div class="entry empty">No hazards encountered yet.</div>`;
    return;
  }
  const iconFor = { data: "⛔", control: "⚠", forward: "↪" };
  log.innerHTML = visible.map(e =>
    `<div class="entry ${e.type}"><span class="entry-icon">${iconFor[e.type]||"•"}</span><div><b>Cycle ${e.cycle}</b><div class="entry-msg">${escapeHtml(e.message)}</div></div></div>`
  ).reverse().join("");
}

function renderStats(upToCycle, instructions, sched, totalCycles) {
  document.getElementById("statInstr").textContent = instructions.length;
  document.getElementById("statCycles").textContent = Math.min(upToCycle, totalCycles);
  document.getElementById("statStalls").textContent = stallCount;
  const cpi = instructions.length ? (totalCycles / instructions.length).toFixed(2) : "0.00";
  document.getElementById("statCPI").textContent = upToCycle >= totalCycles ? cpi : "…";
  const idealCycles = instructions.length + 4;
  const speedup = upToCycle >= totalCycles && totalCycles > 0 ? (totalCycles / idealCycles) : null;
  const effEl = document.getElementById("statEff");
  if (effEl) effEl.textContent = (upToCycle >= totalCycles && instructions.length)
    ? Math.round((idealCycles / totalCycles) * 100) + "%" : "…";
}

function buildRegisterGrid() {
  const grid = document.getElementById("regGrid");
  grid.innerHTML = "";
  for (let i = 0; i < 32; i++) {
    const name = regKey(i);
    const div = document.createElement("div");
    div.className = "reg";
    div.id = "reg-" + name;
    div.innerHTML = `<div class="rname">${name}<span class="abi">${ABI_NAMES[i]}</span></div><div class="rval">${registers[name]}</div>`;
    grid.appendChild(div);
  }
}

function updateRegistersForCycle(cycle, instructions, sched) {
  let changed = [];
  instructions.forEach((instr, i) => {
    if (sched[i].WB === cycle && instr.dest && instr.dest !== "x0") {
      const a = registers[instr.src1] !== undefined ? registers[instr.src1] : 0;
      const b = instr.src2 ? (registers[instr.src2] !== undefined ? registers[instr.src2] : 0) : (parseInt(instr.imm, 10) || 0);
      let result;
      switch (instr.op) {
        case "add": case "addi": result = a + b; break;
        case "sub": result = a - b; break;
        case "and": case "andi": result = a & b; break;
        case "or": case "ori": result = a | b; break;
        case "xor": case "xori": result = a ^ b; break;
        case "sll": result = a << (b & 31); break;
        case "srl": result = a >>> (b & 31); break;
        case "sra": result = a >> (b & 31); break;
        case "slt": case "slti": result = (a < b) ? 1 : 0; break;
        case "mul": result = a * b; break;
        case "div": result = b !== 0 ? Math.trunc(a / b) : 0; break;
        case "lw": case "lh": case "lb": case "ld": result = a + (parseInt(instr.offset,10)||0); break;
        default: result = a;
      }
      registers[instr.dest] = result;
      changed.push(instr.dest);
    }
  });
  changed.forEach(r => {
    const el = document.getElementById("reg-" + r);
    if (el) {
      el.querySelector(".rval").textContent = registers[r];
      el.classList.add("changed");
      setTimeout(() => el.classList.remove("changed"), 800);
    }
  });
}

// ===================================================
// CONTROL LOGIC
// ===================================================
function resetRegisters() {
  initRegisters();
  buildRegisterGrid();
}

function setStatusBadge(text, kind){
  const el = document.getElementById("statusBadge");
  if (!el) return;
  el.textContent = text;
  el.className = "status-badge " + (kind || "");
}

function loadAndRun() {
  const text = document.getElementById("asmInput").value;
  program = parseProgram(text);

  if (program.length === 0) {
    setStatusBadge("No valid instructions found", "err");
    return;
  }

  const result = buildSchedule(program);
  schedule = result.sched;
  totalCycles = result.totalCycles;
  maxCycle = totalCycles;
  currentCycle = 0;

  resetRegisters();
  buildPipelineTable(program, schedule, totalCycles);
  revealUpToCycle(0);
  renderHazardLog(0);
  renderStats(0, program, schedule, totalCycles);
  setStatusBadge(`Loaded ${program.length} instruction${program.length>1?"s":""}`, "ok");

  document.getElementById("stepBtn").disabled = false;
  document.getElementById("playBtn").disabled = false;
  document.getElementById("pauseBtn").disabled = true;
  document.getElementById("cycleReadout").textContent = `Cycle ${0} / ${totalCycles}`;
}

function stepForward() {
  if (currentCycle >= maxCycle) { stopPlaying(); return; }
  currentCycle++;
  revealUpToCycle(currentCycle);
  renderHazardLog(currentCycle);
  renderStats(currentCycle, program, schedule, totalCycles);
  updateRegistersForCycle(currentCycle, program, schedule);
  document.getElementById("cycleReadout").textContent = `Cycle ${currentCycle} / ${maxCycle}`;
  if (currentCycle >= maxCycle) { stopPlaying(); setStatusBadge("Execution complete", "ok"); }
}

function stepBackward() {
  if (currentCycle <= 0) return;
  currentCycle--;
  // recompute registers from scratch up to currentCycle (simplest correct approach)
  initRegisters();
  for (let c = 1; c <= currentCycle; c++) updateRegistersForCycle(c, program, schedule);
  buildRegisterGrid();
  for (let c = 1; c <= currentCycle; c++) updateRegistersForCycle(c, program, schedule);
  revealUpToCycle(currentCycle);
  renderHazardLog(currentCycle);
  renderStats(currentCycle, program, schedule, totalCycles);
  document.getElementById("cycleReadout").textContent = `Cycle ${currentCycle} / ${maxCycle}`;
  document.getElementById("playBtn").disabled = false;
}

function startPlaying() {
  if (currentCycle >= maxCycle) return;
  document.getElementById("playBtn").disabled = true;
  document.getElementById("pauseBtn").disabled = false;
  setStatusBadge("Running…", "run");
  const speed = parseInt(document.getElementById("speedRange").value, 10);
  playTimer = setInterval(() => {
    stepForward();
    if (currentCycle >= maxCycle) stopPlaying();
  }, speed);
}

function stopPlaying() {
  clearInterval(playTimer);
  playTimer = null;
  document.getElementById("playBtn").disabled = (currentCycle >= maxCycle);
  document.getElementById("pauseBtn").disabled = true;
}

function resetAll() {
  stopPlaying();
  currentCycle = 0;
  if (program.length) {
    resetRegisters();
    revealUpToCycle(0);
    renderHazardLog(0);
    renderStats(0, program, schedule, totalCycles);
    document.getElementById("cycleReadout").textContent = `Cycle 0 / ${totalCycles}`;
    document.getElementById("playBtn").disabled = false;
    setStatusBadge(`Loaded ${program.length} instruction${program.length>1?"s":""}`, "ok");
  }
}

// ===================================================
// EXAMPLES — RISC-V
// ===================================================
const EXAMPLES = {
  hazard: `add x5, x6, x7
sub x8, x5, x9
and x10, x8, x11
or  x12, x6, x7`,
  loaduse: `lw  x5, 0(x6)
add x7, x5, x8
sub x9, x7, x10`,
  nohazard: `add x5, x6, x7
sub x8, x9, x10
and x11, x12, x13
or  x14, x15, x16`,
  branch: `add x5, x6, x7
beq x5, x8, LOOP
sub x9, x5, x10
add x11, x9, x5`
};

// ===================================================
// EVENT BINDINGS
// ===================================================
function init(){
  document.getElementById("loadBtn").addEventListener("click", loadAndRun);
  document.getElementById("resetBtn").addEventListener("click", resetAll);
  document.getElementById("stepBtn").addEventListener("click", stepForward);
  document.getElementById("stepBackBtn").addEventListener("click", stepBackward);
  document.getElementById("playBtn").addEventListener("click", startPlaying);
  document.getElementById("pauseBtn").addEventListener("click", stopPlaying);
  document.getElementById("exampleSelect").addEventListener("change", (e) => {
    const val = e.target.value;
    if (val && EXAMPLES[val]) document.getElementById("asmInput").value = EXAMPLES[val];
  });
  resetRegisters();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
