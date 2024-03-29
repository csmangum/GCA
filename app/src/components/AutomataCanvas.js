import React, { useRef, useEffect } from 'react';
import { Box } from '@mui/material';

function AutomataCanvas({ ruleSet }) {
  const canvasRef = useRef(null);
  const cellSize = 10; // size of each cell in pixels
  const gridWidth = 40; // number of generations to display (width)
  const gridHeight = 40; // number of cells in height

  // Convert rule number to binary and use it as the rule set
  const getRuleSet = (ruleNumber) => {
    return ruleNumber.toString(2).padStart(8, '0').split('').map(bit => bit === '1');
  };

  const drawCell = (ctx, x, y, alive) => {
    ctx.fillStyle = alive ? "#000" : "#fff";
    ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
  };

  const updateGrid = (currentGrid, ruleBinary) => {
    const newGrid = Array(gridHeight).fill(false);
    for (let i = 0; i < gridHeight; i++) {
      const leftNeighbor = i === 0 ? currentGrid[gridHeight - 1] : currentGrid[i - 1];
      const rightNeighbor = i === gridHeight - 1 ? currentGrid[0] : currentGrid[i + 1];
      const self = currentGrid[i];
      const ruleIndex = (leftNeighbor ? 4 : 0) + (self ? 2 : 0) + (rightNeighbor ? 1 : 0);
      newGrid[i] = ruleBinary[7 - ruleIndex];
    }
    return newGrid;
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    canvas.width = gridWidth * cellSize;
    canvas.height = gridHeight * cellSize;

    let generations = [Array(gridHeight).fill(false)];
    generations[0][Math.floor(gridHeight / 2)] = true; // Initialize the first generation with a single active cell in the middle

    let animationFrameId;

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      generations.forEach((generation, genIndex) => {
        generation.forEach((cell, cellIndex) => {
          drawCell(ctx, genIndex, cellIndex, cell);
        });
      });

      if (generations.length < gridWidth) {
        generations.push(updateGrid(generations[generations.length - 1], getRuleSet(ruleSet)));
      } else {
        generations.shift();
        generations.push(updateGrid(generations[generations.length - 1], getRuleSet(ruleSet)));
      }

      animationFrameId = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [ruleSet]); // Re-run when ruleSet changes

  return (
    <Box sx={{ border: '1px solid #000', margin: '20px', overflow: 'hidden' }}>
      <canvas ref={canvasRef} />
    </Box>
  );
}

export default AutomataCanvas;
