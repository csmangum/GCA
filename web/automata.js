document.addEventListener("DOMContentLoaded", function () {
  const canvas = document.getElementById("automataCanvas");
  const ctx = canvas.getContext("2d");
  const playPauseButton = document.getElementById("playPauseButton");
  const updateIntervalInput = document.getElementById("updateIntervalInput");
  const ruleSetSelect = document.getElementById("ruleSetSelect");

  // Populate the ruleSetSelect with options
  for (let i = 0; i < 256; i++) {
    const option = document.createElement("option");
    option.value = i;
    option.text = `Rule ${i}`;
    ruleSetSelect.appendChild(option);
  }

  let cellSize = 10,
    gridWidth = 40,
    gridHeight = 40;
  let ruleSet = getRuleSet(parseInt(ruleSetSelect.value));
  let updateInterval = parseInt(updateIntervalInput.value);
  let isPlaying = true; // Animation state
  let lastFrameTime = Date.now();

  canvas.width = gridWidth * cellSize;
  canvas.height = gridHeight * cellSize;

  let generations = [Array(gridHeight).fill(false)];

  function init() {
    generations[0][Math.floor(gridHeight / 2)] = true;
    if (isPlaying) draw();
  }

  function draw() {
    const currentTime = Date.now();
    const timeElapsed = currentTime - lastFrameTime;

    if (!isPlaying) return;

    if (timeElapsed > updateInterval) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      generations.forEach((generation, genIndex) => {
        generation.forEach((cell, cellIndex) => {
          ctx.fillStyle = cell ? "#000" : "#fff";
          ctx.fillRect(
            cellIndex * cellSize,
            genIndex * cellSize,
            cellSize,
            cellSize
          );
        });
      });

      updateGenerations();
      lastFrameTime = currentTime;
    }

    requestAnimationFrame(draw);
  }

  function updateGenerations() {
    if (generations.length < gridWidth) {
      const newGeneration = updateGrid(
        generations[generations.length - 1],
        ruleSet
      );
      generations.push(newGeneration);
    } else {
      generations.shift();
      const newGeneration = updateGrid(
        generations[generations.length - 1],
        ruleSet
      );
      generations.push(newGeneration);
    }
  }

  function getRuleSet(ruleNumber) {
    const ruleBinary = ruleNumber.toString(2).padStart(8, "0");
    return ruleBinary.split("").map((bit) => bit === "1");
  }

  function updateGrid(grid, ruleSet) {
    const newGrid = Array(grid.length).fill(false);
    for (let i = 0; i < grid.length; i++) {
      const leftNeighbor = i === 0 ? grid[grid.length - 1] : grid[i - 1];
      const rightNeighbor = i === grid.length - 1 ? grid[0] : grid[i + 1];
      const self = grid[i];
      const ruleIndex =
        ((leftNeighbor ? 1 : 0) << 2) |
        ((self ? 1 : 0) << 1) |
        (rightNeighbor ? 1 : 0);
      newGrid[i] = ruleSet[7 - ruleIndex];
    }
    return newGrid;
  }

  // Event Listeners
  playPauseButton.addEventListener("click", function () {
    isPlaying = !isPlaying;
    playPauseButton.textContent = isPlaying ? "Pause" : "Play";
    if (isPlaying && !lastFrameTime) {
      // Ensure animation resumes correctly
      lastFrameTime = Date.now();
      draw();
    }
  });

  updateIntervalInput.addEventListener("input", function () {
    updateInterval = parseInt(this.value, 10);
  });

  ruleSetSelect.addEventListener("change", function () {
    ruleSet = getRuleSet(parseInt(this.value, 10));
    // Reset generations on ruleSet change
    generations = [Array(gridHeight).fill(false)];
    generations[0][Math.floor(gridHeight / 2)] = true;
    if (isPlaying) draw();
  });

  init();
});
