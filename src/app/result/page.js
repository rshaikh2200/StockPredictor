'use client';

import { useState } from 'react';
import PropTypes from 'prop-types';
import {
  Container,
  Typography,
  Box,
  Button,
  LinearProgress,
} from '@mui/material';

function ResultPage({ caseStudies }) {
  const [currentPage, setCurrentPage] = useState(0);

  const totalCaseStudies = caseStudies.length;

  const handleNext = () => {
    if (currentPage < totalCaseStudies - 1) {
      setCurrentPage(currentPage + 1);
    }
  };

  const handlePrevious = () => {
    if (currentPage > 0) {
      setCurrentPage(currentPage - 1);
    }
  };

  const completionPercentage = ((currentPage + 1) / totalCaseStudies) * 100;

  return (
    <Container>
      <Typography variant="h4" gutterBottom>
        Case Study {currentPage + 1} of {totalCaseStudies}
      </Typography>

      {/* Display Current Case Study and Questions */}
      <Box mb={3}>
        <Typography variant="h6">Case Study {currentPage + 1}</Typography>
        <Typography variant="body1">{caseStudies[currentPage].summary}</Typography>

        <Typography variant="h6" mt={2}>Questions</Typography>
        <ol>
          {caseStudies[currentPage].questions.map((question, index) => (
            <li key={index}>{question}</li>
          ))}
        </ol>
      </Box>

      {/* Status Bar */}
      <Box mb={3}>
        <LinearProgress variant="determinate" value={completionPercentage} />
        <Typography variant="body2" align="center">
          {Math.round(completionPercentage)}% Completed
        </Typography>
      </Box>

      {/* Navigation Buttons */}
      <Box display="flex" justifyContent="space-between" mb={3}>
        <Button
          variant="contained"
          color="primary"
          disabled={currentPage === 0}
          onClick={handlePrevious}
        >
          Previous
        </Button>
        <Button
          variant="contained"
          color="primary"
          disabled={currentPage === totalCaseStudies - 1}
          onClick={handleNext}
        >
          Next
        </Button>
      </Box>
    </Container>
  );
}

ResultPage.propTypes = {
  caseStudies: PropTypes.arrayOf(
    PropTypes.shape({
      summary: PropTypes.string.isRequired,
      questions: PropTypes.arrayOf(PropTypes.string).isRequired,
    })
  ).isRequired,
};

export default ResultPage;

