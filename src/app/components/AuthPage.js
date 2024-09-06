import React, { useState } from 'react';
import {
  Avatar, Button, CssBaseline, TextField, FormControlLabel, Checkbox, Link,
  Paper, Box, Grid, Typography, CircularProgress, Divider, AppBar, Toolbar
} from '@mui/material';
import LockOutlinedIcon from '@mui/icons-material/LockOutlined';
import GoogleIcon from '@mui/icons-material/Google';
import { signInWithEmailAndPassword, createUserWithEmailAndPassword, signInWithPopup } from 'firebase/auth';
import { googleProvider, auth } from '../firebase';

export default function AuthPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [isSignUp, setIsSignUp] = useState(false);

  const handleAuth = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError('');

    try {
      if (isSignUp) {
        await createUserWithEmailAndPassword(auth, email, password);
      } else {
        await signInWithEmailAndPassword(auth, email, password);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleSignIn = async () => {
    setLoading(true);
    setError('');

    try {
      await signInWithPopup(auth, googleProvider);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Grid container component="main" sx={{ height: '50vh' }}>
      <CssBaseline />

      {/* Header */}
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" sx={{ flexGrow: 1 }}>
            Smart AI Service Guaranteed!
          </Typography>
        </Toolbar>
      </AppBar>

      <Grid
        container
        justifyContent="center"
        alignItems="center"
        sx={{ mt: 4 }} // Adds space between header and form
      >
        <Grid item xs={12} sm={8} md={5} component={Paper} elevation={6} square>
          <Box
            sx={{
              my: 1,
              mx: 3,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
            }}
          >
            <Avatar sx={{ m: 1, bgcolor: 'secondary.main' }}>
              <LockOutlinedIcon />
            </Avatar>
            <Typography component="h1" variant="h5">
              {isSignUp ? 'Create Account' : 'Sign in'}
            </Typography>
            {error && <Typography color="error">{error}</Typography>}
            <Box component="form" noValidate onSubmit={handleAuth} sx={{ mt: 1 }}>
              <TextField
                margin="normal"
                required
                fullWidth
                id="email"
                label="Email Address"
                name="email"
                autoComplete="email"
                autoFocus
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
              <TextField
                margin="normal"
                required
                fullWidth
                name="password"
                label="Password"
                type="password"
                id="password"
                autoComplete="current-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
              <FormControlLabel
                control={<Checkbox value="remember" color="primary" />}
                label="Remember me"
              />
              <Button
                type="submit"
                fullWidth
                variant="contained"
                sx={{ mt: 3, mb: 1 }} // Decreased space below the button
                disabled={loading}
              >
                {loading ? <CircularProgress size={24} /> : isSignUp ? 'Sign Up' : 'Sign In'}
              </Button>
              <Divider sx={{ width: '100%' }}>or</Divider>
              <Button
                variant="outlined"
                fullWidth
                onClick={handleGoogleSignIn}
                disabled={loading}
                startIcon={<GoogleIcon />}
                sx={{ mt: 2 }}
              >
                {loading ? <CircularProgress size={24} /> : 'Sign in with Google'}
              </Button>
              <Grid container sx={{ mt: 2 }}>
                <Grid item>
                  <Link
                    variant="body2"
                    onClick={() => setIsSignUp(!isSignUp)}
                  >
                    {isSignUp ? 'Already have an account? Sign In' : "Don't have an account? Sign Up"}
                  </Link>
                </Grid>
              </Grid>
            </Box>
          </Box>
        </Grid>
      </Grid>
    </Grid>
  );
}