document.addEventListener("DOMContentLoaded", function () {
  const canvas = document.getElementById("automataCanvas");
  const ctx = canvas.getContext("2d");

  // Configuration
  const cellSize = 10; // size of each cell in pixels
  const gridWidth = 40; // number of generations to display (width)
  const gridHeight = 40; // number of cells in height, now represents the 'time' dimension
  const ruleSet = getRuleSet(30); // Default rule
  const updateInterval = 20; // Time between updates in milliseconds

  // Set canvas size based on grid and cell size
  canvas.width = gridWidth * cellSize;
  canvas.height = gridHeight * cellSize;

  let generations = [Array(gridHeight).fill(false)]; // Note: Now each 'generation' represents a vertical column

  function init() {
    // Create the first generation with a single active cell in the middle
    generations[0][Math.floor(gridHeight / 2)] = true;
    draw();
  }

  function draw() {
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw each 'generation' as a vertical column
    generations.forEach((generation, genIndex) => {
      generation.forEach((cell, cellIndex) => {
        ctx.fillStyle = cell ? "#000" : "#fff";
        ctx.fillRect(
          genIndex * cellSize, // Moves horizontally
          cellIndex * cellSize, // Stays as the vertical position
          cellSize,
          cellSize
        );
      });
    });

    // Calculate next generation and add it to the end of the generations array
    if (generations.length < gridWidth) {
      const newGeneration = updateGrid(
        generations[generations.length - 1],
        ruleSet
      );
      generations.push(newGeneration); // Add to the end for rightward growth
    } else {
      // Shift the array to the left to maintain the size and add the new one to the end
      generations.shift(); // Remove the oldest (leftmost) generation
      const newGeneration = updateGrid(
        generations[generations.length - 1],
        ruleSet
      );
      generations.push(newGeneration); // Add the new one to the right
    }

    // Set the timeout for the next draw call
    setTimeout(draw, updateInterval);
  }

  function getRuleSet(ruleNumber) {
    const ruleBinary = ruleNumber.toString(2).padStart(8, "0");
    return ruleBinary.split("").map((bit) => bit === "1");
  }

  function updateGrid(grid, ruleSet) {
    const newGrid = new Array(grid.length).fill(false);
    for (let i = 0; i < grid.length; i++) {
      const leftNeighbor = i === 0 ? grid[grid.length - 1] : grid[i - 1];
      const rightNeighbor = i === grid.length - 1 ? grid[0] : grid[i + 1];
      const self = grid[i];
      const ruleIndex =
        ((leftNeighbor ? 1 : 0) << 2) |
        ((self ? 1 : 0) << 1) |
        (rightNeighbor ? 1 : 0);
      newGrid[i] = ruleSet[7 - ruleIndex]; // RuleSet is used in reverse since 1s represent "alive"
    }
    return newGrid;
  }

  init();
});
