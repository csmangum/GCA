document.addEventListener("DOMContentLoaded", function () {
  const canvas = document.getElementById("automataCanvas");
  const ctx = canvas.getContext("2d");

  // Configuration
  const cellSize = 10;
  const gridWidth = 40;
  const gridHeight = 40;
  const ruleSet = getRuleSet(30);

  canvas.width = gridWidth * cellSize;
  canvas.height = gridHeight * cellSize;

  let generations = [Array(gridHeight).fill(false)];

  function init() {
    generations[0][Math.floor(gridHeight / 2)] = true;
    draw();
  }

  function draw() {
    // Clear canvas at the beginning of each draw call
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    generations.forEach((generation, genIndex) => {
      generation.forEach((cell, cellIndex) => {
        ctx.fillStyle = cell ? "#000" : "#fff";
        ctx.fillRect(genIndex * cellSize, cellIndex * cellSize, cellSize, cellSize);
      });
    });

    updateGenerations(); // Update generations before the next frame
    requestAnimationFrame(draw); // Always call at the end to ensure continuous animation
  }

  function updateGenerations() {
    if (generations.length < gridWidth) {
      const newGeneration = updateGrid(generations[generations.length - 1], ruleSet);
      generations.push(newGeneration);
    } else {
      generations.shift();
      const newGeneration = updateGrid(generations[generations.length - 1], ruleSet);
      generations.push(newGeneration);
    }
  }

  function getRuleSet(ruleNumber) {
    const ruleBinary = ruleNumber.toString(2).padStart(8, "0");
    return ruleBinary.split("").map(bit => bit === "1");
  }

  function updateGrid(grid, ruleSet) {
    const newGrid = new Array(grid.length).fill(false);
    for (let i = 0; i < grid.length; i++) {
      const leftNeighbor = i === 0 ? grid[grid.length - 1] : grid[i - 1];
      const rightNeighbor = i === grid.length - 1 ? grid[0] : grid[i + 1];
      const self = grid[i];
      const ruleIndex = ((leftNeighbor ? 1 : 0) << 2) | ((self ? 1 : 0) << 1) | (rightNeighbor ? 1 : 0);
      newGrid[i] = ruleSet[7 - ruleIndex];
    }
    return newGrid;
  }

  init();
});
