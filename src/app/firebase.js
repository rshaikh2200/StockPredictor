// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyBgrydVwKcI02lLPRu_jk51n7PitVl4lRw",
  authDomain: "flashcard-saas-ca81b.firebaseapp.com",
  projectId: "flashcard-saas-ca81b",
  storageBucket: "flashcard-saas-ca81b.appspot.com",
  messagingSenderId: "191295278710",
  appId: "1:191295278710:web:cf651a7d75cc5133949957"
};

const app = initializeApp(firebaseConfig);

const db = getFirestore(app);

export { db };