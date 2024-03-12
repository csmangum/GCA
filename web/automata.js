document.addEventListener("DOMContentLoaded", function () {
  const canvas = document.getElementById("automataCanvas");
  const ctx = canvas.getContext("2d");
  const playPauseButton = document.getElementById("playPauseButton");
  const animationSpeedSlider = document.getElementById("animationSpeedSlider");
  const gridSizeInput = document.getElementById("gridSizeInput");
  const cellSizeInput = document.getElementById("cellSizeInput");
  const resetButton = document.getElementById("resetButton");
  const ruleSetSelect = document.getElementById("ruleSetSelect");

  // Populate the ruleSetSelect with options
  for (let i = 0; i < 256; i++) {
    const option = document.createElement("option");
    option.value = i;
    option.text = `Rule ${i}`;
    ruleSetSelect.appendChild(option);
  }

  let cellSize = parseInt(cellSizeInput.value, 10),
    gridWidth = parseInt(gridSizeInput.value, 10),
    gridHeight = gridWidth; // Assuming square grid for simplicity
  let ruleSet = getRuleSet(parseInt(ruleSetSelect.value));
  let updateInterval = parseInt(animationSpeedSlider.value);
  let isPlaying = true; // Animation state
  let lastUpdateTime = 0;
  let frameRequest;

  function resizeCanvas() {
    canvas.width = gridWidth * cellSize;
    canvas.height = gridHeight * cellSize;
  }

  function initGenerations() {
    generations = [Array(gridHeight).fill(false)];
    generations[0][Math.floor(gridHeight / 2)] = true;
  }

  let generations = [Array(gridHeight).fill(false)];
  initGenerations();
  resizeCanvas();

  function draw(timestamp) {
    const delay = Math.max(1, 1001 - updateInterval);

    if (!isPlaying) {
      lastUpdateTime = timestamp;
      return;
    }

    if (!lastUpdateTime || timestamp - lastUpdateTime > delay) {
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
      lastUpdateTime = timestamp;
    }

    frameRequest = requestAnimationFrame(draw);
  }

  function updateGenerations() {
    if (generations.length < gridHeight) {
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

  playPauseButton.addEventListener("click", function () {
    isPlaying = !isPlaying;
    playPauseButton.textContent = isPlaying ? "Pause" : "Play";
    if (isPlaying) {
      lastUpdateTime = performance.now();
      frameRequest = requestAnimationFrame(draw);
    } else if (frameRequest) {
      cancelAnimationFrame(frameRequest);
    }
  });

  animationSpeedSlider.addEventListener("input", function () {
    updateInterval = parseInt(this.value, 10);
    lastUpdateTime = performance.now();
  });

  gridSizeInput.addEventListener("input", function () {
    gridWidth = parseInt(this.value, 10);
    gridHeight = gridWidth; // Keep the grid square
    resizeCanvas();
    initGenerations();
    if (isPlaying) draw();
  });

  cellSizeInput.addEventListener("input", function () {
    cellSize = parseInt(this.value, 10);
    resizeCanvas();
    if (isPlaying) draw();
  });

  ruleSetSelect.addEventListener("change", function () {
    ruleSet = getRuleSet(parseInt(this.value, 10));
    initGenerations();
    if (isPlaying) draw();
  });

  resetButton.addEventListener("click", function () {
    initGenerations();
    if (isPlaying) draw();
  });

  // Initialize animation
  if (isPlaying) {
    frameRequest = requestAnimationFrame(draw);
  }
});
