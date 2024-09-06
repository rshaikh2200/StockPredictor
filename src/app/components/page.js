'use client';

import React, { useEffect, useState } from "react";
import { Box, Paper, Typography, Button, Radio, RadioGroup, FormControlLabel, FormControl, FormLabel } from '@mui/material';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';

// Dark mode theme
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#1A73E8',
    },
    secondary: {
      main: '#9C27B0',
    },
    background: {
      default: '#121212',
      paper: '#1D1D1D',
    },
    text: {
      primary: '#E0E0E0',
      secondary: '#B0B0B0',
    },
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
        },
      },
    },
  },
});

export default function Home() {
  const [caseStudies, setCaseStudies] = useState([]);
  
  useEffect(() => {
    // Fetch case studies from your API or static data
    async function fetchCaseStudies() {
      try {
        const response = await fetch('/api/case-studies');  // Adjust API route as needed
        if (!response.ok) throw new Error('Failed to fetch case studies');
        const data = await response.json();
        setCaseStudies(data.caseStudies);
      } catch (error) {
        console.error(error);
      }
    }

    fetchCaseStudies();
  }, []);

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Box
        width="100vw"
        height="100vh"
        display="flex"
        flexDirection="column"
        alignItems="center"
        p={3}
      >
        <Typography variant="h4" gutterBottom sx={{ color: 'text.primary' }}>
          Case Studies with Questions
        </Typography>

        {caseStudies.length === 0 ? (
          <Typography variant="body1" sx={{ color: 'text.secondary' }}>
            Loading case studies...
          </Typography>
        ) : (
          caseStudies.map((caseStudy, index) => (
            <Paper key={index} elevation={6} sx={{ mb: 4, p: 3, width: '100%', maxWidth: '800px', bgcolor: 'background.paper' }}>
              <Typography variant="h6" sx={{ color: 'text.primary' }}>
                Case Study {index + 1}
              </Typography>
              <Typography variant="body1" sx={{ color: 'text.secondary', mt: 2 }}>
                {caseStudy.summary}
              </Typography>

              <FormControl component="fieldset" sx={{ mt: 2 }}>
                <FormLabel component="legend" sx={{ color: 'text.primary' }}>Questions</FormLabel>
                {caseStudy.questions.map((question, qIndex) => (
                  <Box key={qIndex} sx={{ mb: 2 }}>
                    <Typography variant="body2" sx={{ color: 'text.primary' }}>
                      {question}
                    </Typography>
                    <RadioGroup name={`question-${index}-${qIndex}`}>
                      <FormControlLabel value="option1" control={<Radio />} label="Option 1" />
                      <FormControlLabel value="option2" control={<Radio />} label="Option 2" />
                      <FormControlLabel value="option3" control={<Radio />} label="Option 3" />
                    </RadioGroup>
                  </Box>
                ))}
              </FormControl>
            </Paper>
          ))
        )}
      </Box>
    </ThemeProvider>
  );
}
