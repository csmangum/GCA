import React from 'react';
import { Box, Slider, Typography } from '@mui/material';

function ControlPanel({ ruleSet, setRuleSet }) {
  const handleRuleChange = (event, newValue) => {
    setRuleSet(newValue);
  };

  return (
    <Box sx={{ width: 300, margin: '20px' }}>
      <Typography id="input-slider" gutterBottom>
        Rule Set: {ruleSet}
      </Typography>
      <Slider
        aria-labelledby="input-slider"
        value={typeof ruleSet === 'number' ? ruleSet : 30}
        onChange={handleRuleChange}
        min={0}
        max={255}
      />
    </Box>
  );
}

export default ControlPanel;
