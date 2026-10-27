import { FirebaseApp, getApp, getApps, initializeApp } from 'firebase/app';
import { Auth, getAuth } from 'firebase/auth';

export type FirebaseWebConfig = {
  apiKey: string;
  authDomain: string;
  projectId: string;
  storageBucket: string;
  messagingSenderId: string;
  appId: string;
};

function readConfig(): FirebaseWebConfig | null {
  if (process.env.REACT_APP_AUTH_DISABLED?.trim().toLowerCase() === 'true') {
    return null;
  }

  const apiKey = process.env.REACT_APP_FIREBASE_API_KEY?.trim();
  const authDomain = process.env.REACT_APP_FIREBASE_AUTH_DOMAIN?.trim();
  const projectId = process.env.REACT_APP_FIREBASE_PROJECT_ID?.trim();
  const storageBucket = process.env.REACT_APP_FIREBASE_STORAGE_BUCKET?.trim();
  const messagingSenderId = process.env.REACT_APP_FIREBASE_MESSAGING_SENDER_ID?.trim();
  const appId = process.env.REACT_APP_FIREBASE_APP_ID?.trim();

  if (!apiKey || !authDomain || !projectId || !storageBucket || !messagingSenderId || !appId) {
    return null;
  }

  return {
    apiKey,
    authDomain,
    projectId,
    storageBucket,
    messagingSenderId,
    appId,
  };
}

export const firebaseConfig = readConfig();
export const isFirebaseConfigured = firebaseConfig !== null;

let appInstance: FirebaseApp | null = null;
let authInstance: Auth | null = null;

export function getFirebaseApp(): FirebaseApp {
  if (!firebaseConfig) {
    throw new Error('Firebase is not configured. Copy .env.example to .env.local and add your web app keys.');
  }
  if (!appInstance) {
    appInstance = getApps().length ? getApp() : initializeApp(firebaseConfig);
  }
  return appInstance;
}

export function getFirebaseAuth(): Auth {
  if (!authInstance) {
    authInstance = getAuth(getFirebaseApp());
  }
  return authInstance;
}
