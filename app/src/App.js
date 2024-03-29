import React, { useState } from 'react';
import ControlPanel from './components/ControlPanel';
import AutomataCanvas from './components/AutomataCanvas';
import { Box } from '@mui/material';

function App() {
  const [ruleSet, setRuleSet] = useState(30);

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100vh',
        backgroundColor: '#f0f0f0',
      }}
    >
      <ControlPanel ruleSet={ruleSet} setRuleSet={setRuleSet} />
      <AutomataCanvas ruleSet={ruleSet} />
    </Box>
  );
}

export default App;
