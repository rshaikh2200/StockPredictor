"use client";

import { useUser } from "@clerk/nextjs";
import {
    collection,
    doc,
    getDoc,
    getDocs,
    writeBatch,
    setDoc,
} from "firebase/firestore";
import { db } from "../../firebase";  
import {
    Box,
    Button,
    Card,
    CardActionArea,
    CardContent,
    Container,
    Dialog,
    DialogActions,
    DialogContent,
    DialogContentText,
    DialogTitle,
    Grid,
    TextField,
    Typography,
} from "@mui/material";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import EditIcon from '@mui/icons-material/Edit';
import DeleteIcon from '@mui/icons-material/Delete';



export default function Flashcards() {
    const { isLoaded, isSignedIn, user } = useUser();
    const [flashcards, setFlashcards] = useState([]);
    const [name, setName] = useState("");
    const [oldName, setOldName] = useState("");
    const [open, setOpen] = useState(false);
    const router = useRouter();

    useEffect(() => {
        async function getFlashcards() {
            if (!user) return;
            const docRef = doc(db, "users", user.id);
            const docSnap = await getDoc(docRef);

            if (docSnap.exists()) {
                const collections = docSnap.data().flashcards || [];
                setFlashcards(collections);
            } else {
                await setDoc(docRef, { flashcards: [] });
            }
        }
        getFlashcards();
    }, [user]);

    const handleDelete = async (flashcardName) => {
        const updatedFlashcards = flashcards.filter(
            (flashcard) => flashcard.name !== flashcardName,
        );
        const docRef = doc(db, "users", user.id);
        await setDoc(docRef, { flashcards: updatedFlashcards });

        setFlashcards(updatedFlashcards);

        const subColRef = collection(docRef, flashcardName);
        const snapshot = await getDocs(subColRef);
        const batch = writeBatch(db);
        snapshot.docs.forEach((doc) => {
            batch.delete(doc.ref);
        });
        await batch.commit();
    };

    if (!isLoaded || !isSignedIn) return <></>;

    const handleCardClick = (id) => {
        router.push(`/flashcard?id=${id}`);
    };

    const handleOpen = (flashcardName) => {
        setOpen(true);
        setOldName(flashcardName);
    };

    const handleClose = () => {
        setOpen(false);
        setName("");
    };

    const editFlashcard = async (oldName, newName) => {
        if (!newName) {
            alert("Please enter a new name");
            return;
        }

        if (!user) {
            alert("User is not authenticated");
            return;
        }

        const userDocRef = doc(db, "users", user.id);
        const docSnap = await getDoc(userDocRef);

        if (docSnap.exists()) {
            let collections = docSnap.data().flashcards || [];
            const flashcardIndex = collections.findIndex(
                (f) => f.name === oldName,
            );
            if (collections.some((f) => f.name === newName)) {
                alert(
                    "A flashcard with this name already exists. Please choose a different name.",
                );
                return;
            }

            if (flashcardIndex === -1) {
                alert("Flashcard collection not found");
                return;
            }

            collections[flashcardIndex].name = newName;
            await setDoc(
                userDocRef,
                { flashcards: collections },
                { merge: true },
            );
        } else {
            alert("User document does not exist");
            return;
        }

        setFlashcards(
            flashcards.map((flashcard) => {
                if (flashcard.name === oldName) {
                    return { ...flashcard, name: newName };
                }
                return flashcard;
            }),
        );

        const oldSubColRef = collection(userDocRef, oldName);
        const newSubColRef = collection(userDocRef, newName);

        async function copyAndDeleteOldSubCollection() {
            const querySnapshot = await getDocs(oldSubColRef);
            const batch = writeBatch(db);

            querySnapshot.forEach((docSnapshot) => {
                const oldDocRef = docSnapshot.ref;
                const newDocRef = doc(
                    db,
                    `${newSubColRef.path}/${docSnapshot.id}`,
                );
                batch.set(newDocRef, docSnapshot.data());
                batch.delete(oldDocRef);
            });

            await batch.commit();
        }

        copyAndDeleteOldSubCollection()
            .then(() => {
                handleClose();
                console.log("Subcollection name changed successfully.");
            })
            .catch((error) => {
                console.error("Error changing subcollection name: ", error);
            });
    };

    return (
        <Container>
            <Button href="/generate" variant="contained" color="primary">
                Back to Generate
            </Button>
            <Grid container spacing={2}>
                {flashcards.map((flashcard, index) => (
                    <Grid item key={index} xs={12} sm={6} md={4}>
                        <Card sx={{ display: "flex" }}>
                            <CardActionArea
                                onClick={() => handleCardClick(flashcard.name)}
                            >
                                <CardContent>
                                    <Typography variant="h6" component="div">
                                        {flashcard.name}
                                    </Typography>
                                </CardContent>
                            </CardActionArea>
                            <Button>
                                <EditIcon
                                    onClick={() => handleOpen(flashcard.name)}
                                />
                            </Button>
                            <Button>
                                <DeleteIcon
                                    onClick={() => handleDelete(flashcard.name)}
                                />
                            </Button>
                        </Card>
                    </Grid>
                ))}
            </Grid>
            <Dialog open={open} onClose={handleClose}>
                <DialogTitle>Edit Flashcards</DialogTitle>
                <DialogContent>
                    <DialogContentText>
                        Enter a name for the flashcard collection
                    </DialogContentText>
                    <TextField
                        autoFocus
                        margin="dense"
                        label="Collection Name"
                        type="text"
                        fullWidth
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        variant="outlined"
                    />
                </DialogContent>
                <DialogActions>
                    <Button onClick={handleClose} color="primary">
                        Cancel
                    </Button>
                    <Button
                        onClick={() => editFlashcard(oldName, name)}
                        color="primary"
                    >
                        Save
                    </Button>
                </DialogActions>
            </Dialog>
        </Container>
    );
}
