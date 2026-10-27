import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { User, onAuthStateChanged } from 'firebase/auth';
import { getFirebaseAuth, isFirebaseConfigured } from '../lib/firebase';
import { resetDefaultProjectCache } from '../services/projectsApi';

type AuthContextValue = {
  user: User | null;
  loading: boolean;
  configured: boolean;
};

const AuthContext = createContext<AuthContextValue>({
  user: null,
  loading: true,
  configured: false,
});

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const ownerUidRef = useRef<string | null | undefined>(undefined);

  // Resetting the default-project cache cancels any resolution already in
  // flight, and React runs child effects BEFORE the parent's. So Home's first
  // models load starts ensureDefaultProject(), this effect then fired, and the
  // resolution died on "superseded by an auth change" — the list rendered
  // "Couldn't load your models" on every first visit, while Retry worked
  // because by then the epoch had settled. ownerUidRef starts as `undefined`,
  // so the very first observation compared unequal to null and counted as a
  // change.
  //
  // The first observation is not a change: there is no prior identity and
  // nothing is cached under one. The exception is arriving already signed in —
  // a resolution begun before auth resolved ran without a token, so that one
  // still has to be thrown away.
  const noteOwner = useCallback((nextUid: string | null) => {
    const previous = ownerUidRef.current;
    ownerUidRef.current = nextUid;
    const changed = previous === undefined ? nextUid !== null : previous !== nextUid;
    if (changed) resetDefaultProjectCache();
  }, []);

  useEffect(() => {
    if (!isFirebaseConfigured) {
      noteOwner(null);
      setUser(null);
      setLoading(false);
      return undefined;
    }

    const auth = getFirebaseAuth();
    const unsubscribe = onAuthStateChanged(auth, (nextUser) => {
      noteOwner(nextUser?.uid ?? null);
      setUser(nextUser);
      setLoading(false);
    });
    return unsubscribe;
  }, [noteOwner]);

  const value = useMemo(
    () => ({
      user,
      loading,
      configured: isFirebaseConfigured,
    }),
    [user, loading],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  return useContext(AuthContext);
}
