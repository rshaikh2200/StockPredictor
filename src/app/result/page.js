"use client";

import { useUser } from "@clerk/nextjs";
import { useRouter } from "next/router";
import { useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { CircularProgress, Container, Typography, Box, Button } from "@mui/material";

const ResultPage = () => {
   
    const searchParams = useSearchParams();
    const session_id = searchParams.get('session_id');

    const [loading, setLoading] = useState(true);
    const [session, setSession] = useState(null);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchSession = async () => {
            if (!session_id) return;

            try {
                const res = await fetch(`/api/checkout_session?session_id=${session_id}`);
                const data = await res.json();

                if (!res.ok) {
                    setError(data);
                    return;
                }

                setSession(data);
            } catch (error) {
                setError("An error occurred");
            } finally {
                setLoading(false);
            }
        };

        fetchSession();
    }, [session_id]);

    if (loading) {
        return (
            <Container 
                maxWidth="100vw" 
                sx={{
                    textAlign: 'center',
                    marginTop: '20vh',
                }}>
                <CircularProgress />
                <Typography variant="h6">Loading...</Typography>
            </Container>
        );
    }

    if (error) {
        return (
            <Container 
                maxWidth="100vw" 
                sx={{
                    textAlign: 'center',
                    marginTop: '20vh',
                }}>
                <Typography variant="h6" color="error">{error}</Typography>
            </Container>
        );
    }

    return (
        <Container 
            maxWidth="100vw" 
            sx={{ textAlign: 'center', marginTop: '20vh' }}>
            {session?.payment_status === 'paid' ? (
                <>
                    <Typography variant="h6">Payment successful! Your order is being processed.</Typography>
                    <Box sx={{ mt: 2 }}>
                        <Typography variant="h6">Session ID: {session_id}</Typography>
                        <Typography variant="body1">
                            We have received your payment. Your order is being processed. You will receive an email with the details shortly.
                        </Typography>
                    </Box>
                </>
            ) : (
                <>
                    <Typography variant="h6" color="error">Payment Failed or Incomplete</Typography>
                    <Box sx={{ mt: 2 }}>
                        <Typography variant="h6">Session ID: {session_id}</Typography>
                        <Typography variant="body1">
                            It looks like your payment was not completed. Please try the payment again or contact support if you need help.
                        </Typography>
                        <Button 
                            variant="outlined" 
                            color="primary" 
                            href="/" 
                            sx={{ mt: 2, mr:5}}>
                            Back to Home
                        </Button>
                        {/* <Button 
                            variant="contained" 
                            color="primary" 
                            href="/" // Replace with your checkout URL
                            sx={{ mt: 2 }}>
                            Retry Payment
                        </Button> */}
                    </Box>
                </>
            )}
        </Container>
    );
}

export default ResultPage;