document.addEventListener('DOMContentLoaded', function() {
    const canvas = document.getElementById('automataCanvas');
    const ctx = canvas.getContext('2d');

    // Configuration
    const cellSize = 10; // size of each cell in pixels
    const gridWidth = 50; // number of cells in width
    const gridHeight = 30; // number of generations to display (height)
    const ruleSet = getRuleSet(30); // Default rule

    // Set canvas size based on grid and cell size
    canvas.width = gridWidth * cellSize;
    canvas.height = gridHeight * cellSize;

    let generations = [Array(gridWidth).fill(false)];

    function init() {
        // Create the first generation with a single active cell in the middle
        generations[0][Math.floor(gridWidth / 2)] = true;

        draw();
    }

    function draw() {
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw each generation
        generations.forEach((generation, genIndex) => {
            generation.forEach((cell, cellIndex) => {
                ctx.fillStyle = cell ? '#000' : '#fff';
                ctx.fillRect(cellIndex * cellSize, genIndex * cellSize, cellSize, cellSize);
            });
        });

        // Calculate next generation and add it to the start of the generations array
        if (generations.length < gridHeight) {
            const newGeneration = updateGrid(generations[0], ruleSet);
            generations.unshift(newGeneration);
        } else {
            // Remove the oldest generation to maintain the size and add the new one
            generations.pop();
            const newGeneration = updateGrid(generations[0], ruleSet);
            generations.unshift(newGeneration);
        }

        requestAnimationFrame(draw);
    }

    function getRuleSet(ruleNumber) {
        const ruleBinary = ruleNumber.toString(2).padStart(8, '0');
        return ruleBinary.split('').map(bit => bit === '1');
    }

    function updateGrid(grid, ruleSet) {
        const newGrid = new Array(grid.length).fill(false);
        for (let i = 0; i < grid.length; i++) {
            const leftNeighbor = i === 0 ? grid[grid.length - 1] : grid[i - 1];
            const rightNeighbor = i === grid.length - 1 ? grid[0] : grid[i + 1];
            const self = grid[i];
            const ruleIndex = ((leftNeighbor ? 1 : 0) << 2) | ((self ? 1 : 0) << 1) | (rightNeighbor ? 1 : 0);
            newGrid[i] = ruleSet[7 - ruleIndex]; // RuleSet is used in reverse since 1s represent "alive"
        }
        return newGrid;
    }

    init();
});
