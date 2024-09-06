'use client';

import React, { useState } from 'react';
import { Button, Container, Typography, Box, Dialog, DialogActions, DialogContent, DialogTitle, TextField } from '@mui/material';
import { useRouter } from 'next/navigation'; // Import the useRouter hook
import Head from "next/head";
import Appbar from "/src/app/components/Appbar.jsx"; // Assuming Appbar component exists here

export default function HomePage() {
  // State to control the visibility of the dialog
  const [open, setOpen] = useState(false);

  // State to capture the form inputs
  const [role, setRole] = useState('');
  const [specialty, setSpecialty] = useState('');
  const [department, setDepartment] = useState('');

  // Initialize the router
  const router = useRouter();

  // Function to handle opening the dialog
  const handleOpen = () => {
    setOpen(true);
  };

  // Function to handle closing the dialog
  const handleClose = () => {
    setOpen(false);
  };

  // Function to handle form submission
  const handleSubmit = () => {
    // You can add your logic here to handle the form data
    console.log('Role:', role);
    console.log('Specialty:', specialty);
    console.log('Department:', department);
    
    // Close the dialog after form submission
    handleClose();
    
    // Redirect to the generate page
    router.push('/generate/page.js'); // Redirects to the /generate/page.js route
  };

  return (
    <Container 
      maxWidth="sm" 
      sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', height: '100vh', justifyContent: 'center' }}
    >
      <Head>
        <title>Welcome to Our Platform</title>
      </Head>
      
      <Appbar /> {/* Assuming you have an Appbar component */}

      <Box sx={{ textAlign: 'center', mb: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom>
          Welcome to Our Platform
        </Typography>
        <Typography variant="subtitle1" gutterBottom>
          Get started in testing your knowledge on handling critical workplace situations
        </Typography>
      </Box>

      <Box sx={{ display: 'flex', gap: 2 }}>
        <Button variant="contained" color="primary" size="large">
          Sign In
        </Button>
        <Button variant="outlined" color="secondary" size="large" onClick={handleOpen}>
          Get Started
        </Button>
      </Box>

      {/* Dialog for Get Started Form */}
      <Dialog open={open} onClose={handleClose}>
        <DialogTitle>Enter Your Details</DialogTitle>
        <DialogContent>
          <Box component="form" sx={{ mt: 2 }}>
            <TextField
              label="Role"
              fullWidth
              margin="normal"
              value={role}
              onChange={(e) => setRole(e.target.value)}
            />
            <TextField
              label="Specialty"
              fullWidth
              margin="normal"
              value={specialty}
              onChange={(e) => setSpecialty(e.target.value)}
            />
            <TextField
              label="Department"
              fullWidth
              margin="normal"
              value={department}
              onChange={(e) => setDepartment(e.target.value)}
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleClose} color="secondary">Cancel</Button>
          <Button onClick={handleSubmit} color="primary">Submit</Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
}
