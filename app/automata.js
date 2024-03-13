document.addEventListener("DOMContentLoaded", function () {
  const canvas = document.getElementById("automataCanvas");
  const ctx = canvas.getContext("2d");
  const playPauseButton = document.getElementById("playPauseButton");
  const animationSpeedSlider = document.getElementById("animationSpeedSlider");
  const gridSizeInput = document.getElementById("gridSizeInput");
  const cellSizeInput = document.getElementById("cellSizeInput");
  const resetButton = document.getElementById("resetButton");
  const ruleSetSelect = document.getElementById("ruleSetSelect");
  const ageColorCodingCheckbox = document.getElementById(
    "ageColorCodingCheckbox"
  );
  const colorPaletteSelect = document.getElementById("colorPaletteSelect");

  const colorPalettes = {
    heat: [
      "#ffffcc",
      "#ffeda0",
      "#fed976",
      "#feb24c",
      "#fd8d3c",
      "#fc4e2a",
      "#e31a1c",
      "#bd0026",
      "#800026",
    ],
    rainbow: [
      "#9400D3",
      "#4B0082",
      "#0000FF",
      "#00FF00",
      "#FFFF00",
      "#FF7F00",
      "#FF0000",
    ],
    grayscale: [
      "#ffffff",
      "#e0e0e0",
      "#c0c0c0",
      "#a0a0a0",
      "#808080",
      "#606060",
      "#404040",
      "#202020",
      "#000000",
    ],
  };

  Object.keys(colorPalettes).forEach((key) => {
    const option = document.createElement("option");
    option.value = key;
    option.textContent = key.charAt(0).toUpperCase() + key.slice(1); // Capitalize the name
    colorPaletteSelect.appendChild(option);
  });

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
    generations = [
      Array(gridHeight)
        .fill(null)
        .map(() => ({ alive: false, age: 0 })),
    ];
    generations[0][Math.floor(gridHeight / 2)] = { alive: true, age: 1 };
  }

  let generations = [];
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

      const palette = colorPalettes[colorPaletteSelect.value];

      generations.forEach((generation, genIndex) => {
        generation.forEach((cell, cellIndex) => {
          if (cell.alive) {
            if (ageColorCodingCheckbox.checked) {
              const colorIndex = Math.min(cell.age - 1, palette.length - 1);
              ctx.fillStyle = palette[colorIndex];
            } else {
              ctx.fillStyle = "#000"; // Default color for living cells
            }
          } else {
            ctx.fillStyle = "#fff"; // Color for dead cells
          }
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
    const newGrid = Array(grid.length)
      .fill(null)
      .map(() => ({ alive: false, age: 0 }));
    for (let i = 0; i < grid.length; i++) {
      const leftNeighbor = i === 0 ? grid[grid.length - 1] : grid[i - 1];
      const rightNeighbor = i === grid.length - 1 ? grid[0] : grid[i + 1];
      const self = grid[i];
      const ruleIndex =
        ((leftNeighbor.alive ? 1 : 0) << 2) |
        ((self.alive ? 1 : 0) << 1) |
        (rightNeighbor.alive ? 1 : 0);
      const alive = ruleSet[7 - ruleIndex];
      newGrid[i] = {
        alive: alive,
        age: alive ? (self.alive ? self.age + 1 : 1) : 0,
      };
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
