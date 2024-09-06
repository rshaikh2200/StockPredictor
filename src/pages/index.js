import React from 'react';
import { auth } from '../app/firebase'; // Corrected the import path
import AuthPage from '../app/components/AuthPage'; // Corrected the import path
import Home from '../app/components/page'; // Ensure correct import path
import { useAuthState } from '../app/hooks/useAuthState';

export default function Index() {
  const { user } = useAuthState();

  return user ? <Home /> : <AuthPage />;
}