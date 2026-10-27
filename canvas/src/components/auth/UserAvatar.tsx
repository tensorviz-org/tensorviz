import React, { useEffect, useState } from 'react';
import type { User } from 'firebase/auth';

export function userInitials(displayName?: string | null, email?: string | null): string {
  if (displayName?.trim()) {
    const parts = displayName.trim().split(/\s+/);
    if (parts.length >= 2) return `${parts[0][0]}${parts[1][0]}`.toUpperCase();
    return parts[0].slice(0, 2).toUpperCase();
  }
  if (email) return email.slice(0, 2).toUpperCase();
  return '?';
}

/** Prefer Firebase user photo, then Google provider photo. */
export function resolvePhotoURL(user: User): string | null {
  if (user.photoURL) return user.photoURL;
  const google = user.providerData.find((p) => p.providerId === 'google.com');
  return google?.photoURL ?? null;
}

type UserAvatarProps = {
  user: User;
  className?: string;
};

/**
 * Google profile images often 403 without referrerPolicy="no-referrer".
 * Falls back to initials when the image fails to load.
 */
export default function UserAvatar({ user, className = 'ui2-avatar' }: UserAvatarProps) {
  const photoURL = resolvePhotoURL(user);
  const [showPhoto, setShowPhoto] = useState(Boolean(photoURL));

  useEffect(() => {
    setShowPhoto(Boolean(photoURL));
  }, [photoURL]);

  if (photoURL && showPhoto) {
    return (
      <span className={`${className} overflow-hidden p-0`}>
        <img
          src={photoURL}
          alt=""
          referrerPolicy="no-referrer"
          decoding="async"
          onError={() => setShowPhoto(false)}
          className="w-full h-full object-cover"
        />
      </span>
    );
  }

  return <span className={className}>{userInitials(user.displayName, user.email)}</span>;
}
