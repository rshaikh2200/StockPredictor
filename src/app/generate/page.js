'use client'

import { useState } from 'react'
import {
  Container,
  TextField,
  Button,
  Typography,
  Box,
} from '@mui/material'

export default function Generate() {
  const [role, setRole] = useState('')
  const [specialty, setSpecialty] = useState('')
  const [department, setDepartment] = useState('')
  const [scenarios, setScenarios] = useState([])

  const handleSubmit = async () => {
    if (!role.trim() || !specialty.trim() || !department.trim()) {
      alert('Please enter your role, specialty, and department to generate case studies.')
      return
    }

    try {
      const response = await fetch('/api/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ role, specialty, department }),
      })

      if (!response.ok) {
        throw new Error('Failed to generate case studies and questions.')
      }

      const data = await response.json()
      setScenarios(data.caseStudies)
    } catch (error) {
      console.error('Error generating case studies:', error)
      alert('An error occurred while generating case studies. Please try again.')
    }
  }

  return (
    <Container>
      <Typography variant="h4" gutterBottom>
        Generate Case Studies and Questions
      </Typography>
      <Box mb={2}>
        <TextField
          label="Role"
          value={role}
          onChange={(e) => setRole(e.target.value)}
          fullWidth
          margin="normal"
        />
        <TextField
          label="Specialty"
          value={specialty}
          onChange={(e) => setSpecialty(e.target.value)}
          fullWidth
          margin="normal"
        />
        <TextField
          label="Department"
          value={department}
          onChange={(e) => setDepartment(e.target.value)}
          fullWidth
          margin="normal"
        />
      </Box>
      <Button variant="contained" color="primary" onClick={handleSubmit}>
        Generate Scenarios
      </Button>

      {scenarios.length > 0 && (
        <Box mt={4}>
          <Typography variant="h5">Generated Scenarios</Typography>
          {scenarios.map((scenario, index) => (
            <Box key={index} mb={3}>
              <Typography variant="h6">Case Study {index + 1}</Typography>
              <Typography variant="body1">{scenario.summary}</Typography>
              <Typography variant="h6">Questions</Typography>
              <ol>
                {scenario.questions.map((question, qIndex) => (
                  <li key={qIndex}>{question}</li>
                ))}
              </ol>
            </Box>
          ))}
        </Box>
      )}
    </Container>
  )
}

