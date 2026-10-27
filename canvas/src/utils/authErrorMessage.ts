import { FirebaseError } from 'firebase/app';

const MESSAGES: Record<string, string> = {
  'auth/email-already-in-use': 'An account already exists with this email.',
  'auth/invalid-email': 'Enter a valid email address.',
  'auth/operation-not-allowed': 'This sign-in method is not enabled. Enable it in the Firebase console.',
  'auth/weak-password': 'Password must be at least 6 characters.',
  'auth/user-disabled': 'This account has been disabled.',
  'auth/user-not-found': 'No account found with this email.',
  'auth/wrong-password': 'Incorrect email or password.',
  'auth/invalid-credential': 'Incorrect email or password.',
  'auth/too-many-requests': 'Too many attempts. Wait a moment and try again.',
  'auth/popup-closed-by-user': 'Sign-in was cancelled.',
  'auth/cancelled-popup-request': 'Sign-in was cancelled.',
  'auth/account-exists-with-different-credential':
    'An account already exists with the same email but a different sign-in method.',
  'auth/network-request-failed': 'Network error. Check your connection and try again.',
};

export function authErrorMessage(error: unknown): string {
  if (error instanceof FirebaseError && MESSAGES[error.code]) {
    return MESSAGES[error.code];
  }
  if (error instanceof Error && error.message) {
    return error.message;
  }
  return 'Something went wrong. Please try again.';
}
