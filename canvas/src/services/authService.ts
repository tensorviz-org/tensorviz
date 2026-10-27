import {
  createUserWithEmailAndPassword,
  GoogleAuthProvider,
  sendPasswordResetEmail,
  signInWithEmailAndPassword,
  signInWithPopup,
  signOut,
  updateProfile,
  User,
} from 'firebase/auth';
import { getFirebaseAuth, isFirebaseConfigured } from '../lib/firebase';
import { resetDefaultProjectCache } from './projectsApi';

const googleProvider = new GoogleAuthProvider();

export function assertAuthReady(): void {
  if (!isFirebaseConfigured) {
    throw new Error('Firebase is not configured. Add REACT_APP_FIREBASE_* keys to .env.local.');
  }
}

export async function signInWithEmail(email: string, password: string): Promise<User> {
  assertAuthReady();
  const { user } = await signInWithEmailAndPassword(getFirebaseAuth(), email.trim(), password);
  return user;
}

export async function signUpWithEmail(
  email: string,
  password: string,
  displayName?: string,
): Promise<User> {
  assertAuthReady();
  const { user } = await createUserWithEmailAndPassword(getFirebaseAuth(), email.trim(), password);
  if (displayName?.trim()) {
    await updateProfile(user, { displayName: displayName.trim() });
  }
  return user;
}

export async function signInWithGoogle(): Promise<User> {
  assertAuthReady();
  const { user } = await signInWithPopup(getFirebaseAuth(), googleProvider);
  return user;
}

export async function sendResetPassword(email: string): Promise<void> {
  assertAuthReady();
  await sendPasswordResetEmail(getFirebaseAuth(), email.trim());
}

export async function signOutUser(): Promise<void> {
  assertAuthReady();
  resetDefaultProjectCache();
  await signOut(getFirebaseAuth());
}
