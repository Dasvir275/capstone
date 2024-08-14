import { initializeApp } from 'firebase/app';
import { getAuth } from 'firebase/auth';
import { getFirestore } from 'firebase/firestore';
import { getStorage } from 'firebase/storage';

const firebaseConfig = {
  apiKey: "AIzaSyC-FnLwr7dDq1W5fLQTIc-M6wV2xYRsPug",
  authDomain: "sih1289-ecoprotectors.firebaseapp.com",
  projectId: "sih1289-ecoprotectors",
  storageBucket: "sih1289-ecoprotectors.appspot.com",
  messagingSenderId: "599056463339",
  appId: "1:599056463339:web:e18299b2d030f8468d55ae",
  measurementId: "G-BZB7MPJ68E"
};

const firebaseApp = initializeApp(firebaseConfig);

const auth = getAuth(firebaseApp);
const db = getFirestore(firebaseApp);
const storage = getStorage(firebaseApp);

export { auth, db, storage };
