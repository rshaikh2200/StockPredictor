'use client';
import { Bookmark, BookmarkBorder, Delete } from '@mui/icons-material';
import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useUser } from '@clerk/nextjs';
import {
    IconButton, Container, Box,AppBar,Toolbar,Link, Typography, Button, Grid, Card, CardActionArea, CardContent,
    Dialog, DialogTitle, DialogContent, DialogContentText, DialogActions, TextField, Tabs, Tab
} from '@mui/material';
import { doc, collection, writeBatch, getDoc } from 'firebase/firestore';
import { db } from '@/firebase';
import HomeIcon from '@mui/icons-material/Home';

const gradients = [
    'linear-gradient(135deg, #333333 0%, #104EB1 100%)',
    'linear-gradient(135deg, #333333 0%, #104EB1 100%)',
    'linear-gradient(135deg, #333333 0%, #104EB1 100%)',
    'linear-gradient(135deg, #333333 0%, #104EB1 100%)',
    'linear-gradient(135deg, #333333 0%, #104EB1 100%)'
];

export default function Generate() {
    const [savedCards, setSavedCards] = useState([]);
    const { isLoaded, isSignedIn, user } = useUser();
    const [flashcards, setFlashcards] = useState([]);
    const [flipped, setFlipped] = useState([]);
    const [open, setOpen] = useState(false);
    const [name, setName] = useState('');
    const [text, setText] = useState('');
    const [selectedTab, setSelectedTab] = useState(0);
    const [previewShown, setPreviewShown] = useState(false);
    const router = useRouter();

    const handleSubmit = async () => {
        try {
            const response = await fetch('api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text }),
            });
            if (!response.ok) throw new Error('Network response was not ok');
            const data = await response.json();
            setFlashcards(data);
            setPreviewShown(true);
        } catch (error) {
            console.error('There was a problem with the fetch operation:', error);
        }
    };

    const handleTabChange = (event, newValue) => setSelectedTab(newValue);
    const handleCardClick = (id) => setFlipped(prev => ({ ...prev, [id]: !prev[id] }));
    const handleOpen = () => setOpen(true);
    const handleClose = () => setOpen(false);

    const handleSaveCard = (index) => {
        const cardToSave = flashcards[index];
        setSavedCards(prev => prev.includes(cardToSave) ? prev.filter(card => card !== cardToSave) : [...prev, cardToSave]);
    };

    const saveFlashCards = async () => {
        if (!name) return alert('Please enter the topic');
        const batch = writeBatch(db);
        const userDocRef = doc(collection(db, 'users'), user.id);
        const docSnap = await getDoc(userDocRef);
        const collections = docSnap.exists() ? docSnap.data().flashcards || [] : [];
        if (collections.some(f => f.name === name)) return alert('Flashcard collection with the same name already exists');
        collections.push({ name });
        batch.set(userDocRef, { flashcards: collections }, { merge: true });
        flashcards.forEach(flashcard => batch.set(doc(collection(userDocRef, name)), flashcard));
        await batch.commit();
        handleClose();
        router.push('/flashcards');
    };

    const handleDeleteCard = (index) => setSavedCards(prev => prev.filter((_, i) => i !== index));

    return (
        <Box
            sx={{
                position: 'relative',
                minHeight: '100vh',
                backgroundColor: 'black',
                color: 'white',
                padding: 0,
                '&::before': {
                    content: '""',
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: '100%',
                    backgroundImage: 'linear-gradient(rgba(0, 0, 0, 0.8), rgba(0, 0, 0, 0.5))',
                    backgroundSize: 'cover',
                    backgroundPosition: 'center',
                    backgroundRepeat: 'no-repeat',
                    opacity: 0.9,
                    zIndex: -1,
                }
            }}
        >
          <AppBar position="static" sx={{ backgroundColor: '#000000' }}>
      <Toolbar>
        <Button href="/" sx={{ color: '#ffffff', display: 'flex', alignItems: 'center' }}>
          <HomeIcon fontSize="large" sx={{ mr: 1 }} />
          <Typography variant="h6" component="div">
            Home
          </Typography>
        </Button>
      </Toolbar>
    </AppBar>
            <Container maxWidth="md">
                <Box sx={{ mb: 6, textAlign: 'center' }}>
                    <Typography variant="h4" sx={{ paddingTop: 4, fontWeight: 'bold' }} gutterBottom>
                        Create Flashcards
                    </Typography>
                    <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, maxWidth: 600, mx: 'auto', mt: 4 }}>
                        <TextField
                            value={text}
                            onChange={(e) => setText(e.target.value)}
                            label="Enter title to generate flashcard"
                            variant="outlined"
                            size="medium"
                            sx={{
                                flex: 1,
                                minWidth: 200,
                                borderRadius: '50px',
                                backgroundColor: 'rgba(255, 255, 255, 0.1)',
                                '& .MuiOutlinedInput-root': {
                                    '& fieldset': {
                                        borderColor: 'grey.800',
                                    },
                                    '&:hover fieldset': {
                                        borderColor: 'grey.800',
                                    },
                                    '&.Mui-focused fieldset': {
                                        borderColor: '#4a90e2',
                                    },
                                },
                                '& .MuiInputLabel-root': {
                                    color: 'white',
                                    '&.Mui-focused': {
                                        color: '#4a90e2',
                                    },
                                },
                                '& .MuiInputBase-root': {
                                    borderRadius: '50px',
                                    backgroundColor: 'rgba(255, 255, 255, 0.1)',
                                },
                            }}
                            InputProps={{
                                sx: {
                                    color: 'white',
                                }
                            }}
                        />
                        <Button
                            variant="contained"
                            sx={{
                                borderRadius: 2,
                                backgroundColor: '#104EB1',
                                '&:hover': {
                                    backgroundColor: '#104EB1',
                                }
                            }}
                            onClick={handleSubmit}
                        >
                            Generate
                        </Button>
                    </Box>
                </Box>

                <Box
                    sx={{
                        width: '100%',
                        maxWidth: '1800px',
                        backgroundColor: 'transparent',
                        padding: 3,
                        margin: 'auto',
                        transition: 'box-shadow 0.3s ease',
                        '&:hover': {
                            boxShadow: '0 6px 12px rgba(255, 255, 255, 0.2), 0 12px 24px rgba(255, 255, 255, 0.2)',
                        },
                    }}
                >
                    <Tabs value={selectedTab} onChange={handleTabChange} centered>
                        <Tab label="Preview" sx={{ minWidth: 'unset', color: selectedTab === 0 ? '#4a90e2' : 'white', fontWeight: selectedTab === 0 ? 'bold' : 'normal' }} />
                        <Tab label="Saved Cards" sx={{ minWidth: 'unset', color: selectedTab === 1 ? '#4a90e2' : 'white', fontWeight: selectedTab === 1 ? 'bold' : 'normal' }} />
                    </Tabs>

                    {selectedTab === 0 && (
                        <Box sx={{ p: 3 }}>
                            {flashcards.length > 0 ? (
                                <Grid container spacing={2}>
                                    {flashcards.map((flashcard, index) => (
                                        <Grid item xs={12} sm={6} md={4} key={index}>
                                            <Card
                                                sx={{
                                                    height: 250,
                                                    display: 'flex',
                                                    alignItems: 'center',
                                                    justifyContent: 'center',
                                                    borderRadius: 4,
                                                    background: gradients[index % gradients.length],
                                                    boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1), 0 6px 20px rgba(0, 0, 0, 0.1)',
                                                    transition: 'transform 0.3s ease',
                                                    '&:hover': {
                                                        transform: 'scale(1.05)',
                                                    },
                                                }}
                                            >
                                                <CardActionArea onClick={() => handleCardClick(index)}>
                                                    <CardContent>
                                                        <Typography variant="h6" component="div" color="white">
                                                            {flashcard.front}
                                                        </Typography>
                                                        {flipped[index] && (
                                                            <Typography variant="body2" color="white">
                                                                {flashcard.back}
                                                            </Typography>
                                                        )}
                                                    </CardContent>
                                                </CardActionArea>
                                                <IconButton
                                                    onClick={() => handleSaveCard(index)}
                                                    sx={{ position: 'absolute', top: 8, right: 8 }}
                                                >
                                                    {savedCards.includes(flashcard) ? <Bookmark /> : <BookmarkBorder />}
                                                </IconButton>
                                            </Card>
                                        </Grid>
                                    ))}
                                </Grid>
                            ) : (
                                <Typography variant="body1" color="white">
                                    No flashcards available. Generate some to preview.
                                </Typography>
                            )}
                        </Box>
                    )}

                    {selectedTab === 1 && (
                        <Box sx={{ p: 3 }}>
                            {savedCards.length > 0 ? (
                                <Grid container spacing={2}>
                                    {savedCards.map((flashcard, index) => (
                                        <Grid item xs={12} sm={6} md={4} key={index}>
                                            <Card
                                                sx={{
                                                    height: 250,
                                                    display: 'flex',
                                                    alignItems: 'center',
                                                    justifyContent: 'center',
                                                    borderRadius: 4,
                                                    background: '#0044cc',
                                                    boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1), 0 6px 20px rgba(0, 0, 0, 0.1)',
                                                    transition: 'transform 0.3s ease',
                                                    '&:hover': {
                                                        transform: 'scale(1.05)',
                                                    },
                                                }}
                                            >
                                                <CardActionArea>
                                                    <CardContent>
                                                        <Typography variant="h6" component="div" color="white">
                                                            {flashcard.front}
                                                        </Typography>
                                                        <Typography variant="body2" color="white">
                                                            {flashcard.back}
                                                        </Typography>
                                                    </CardContent>
                                                </CardActionArea>
                                                <IconButton
                                                    onClick={() => handleDeleteCard(index)}
                                                    sx={{ position: 'absolute', top: 8, right: 8 }}
                                                >
                                                    <Delete />
                                                </IconButton>
                                            </Card>
                                        </Grid>
                                    ))}
                                </Grid>
                            ) : (
                                <Typography variant="body1" color="white">
                                    No saved flashcards yet.
                                </Typography>
                            )}
                        </Box>
                    )}
                </Box>

                <Button
                    variant="outlined"
                    sx={{
                        borderRadius: 2,
                        borderColor: '#4a90e2',
                        color: '#4a90e2',
                        '&:hover': {
                            borderColor: '#357abd',
                            color: '#357abd',
                        },
                        mt: 4,
                    }}
                    onClick={handleOpen}
                >
                    Save Flashcards
                </Button>

                <Dialog open={open} onClose={handleClose}>
                    <DialogTitle>Save Flashcards</DialogTitle>
                    <DialogContent>
                        <DialogContentText>
                            Enter a name for your flashcard collection.
                        </DialogContentText>
                        <TextField
                            autoFocus
                            margin="dense"
                            label="Collection Name"
                            type="text"
                            fullWidth
                            variant="outlined"
                            value={name}
                            onChange={(e) => setName(e.target.value)}
                            sx={{
                                '& .MuiOutlinedInput-root': {
                                    '& fieldset': {
                                        borderColor: 'grey.800',
                                    },
                                    '&:hover fieldset': {
                                        borderColor: 'grey.800',
                                    },
                                    '&.Mui-focused fieldset': {
                                        borderColor: '#4a90e2',
                                    },
                                },
                                '& .MuiInputLabel-root': {
                                    color: 'white',
                                    '&.Mui-focused': {
                                        color: '#4a90e2',
                                    },
                                },
                            }}
                        />
                    </DialogContent>
                    <DialogActions>
                        <Button onClick={handleClose}>Cancel</Button>
                        <Button onClick={saveFlashCards}>Save</Button>
                    </DialogActions>
                </Dialog>
            </Container>
        </Box>
    );
}