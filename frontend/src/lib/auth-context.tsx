import { createContext, useContext, useEffect, useState, type ReactNode } from "react";

export interface User {
  name: string;
  email: string;
  provider?: "email" | "google";
}

interface AuthContextType {
  user: User | null;
  isLoggedIn: boolean;
  login: (email: string, name?: string) => void;
  signup: (name: string, email: string) => void;
  loginWithGoogle: () => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

const STORAGE_KEY = "earth_query_lens_user";

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        setUser(JSON.parse(saved));
      }
    } catch {
      // ignore JSON error
    }
  }, []);

  const login = (email: string, name?: string) => {
    const derivedName = name || email.split("@")[0] || "User";
    const newUser: User = { email, name: derivedName, provider: "email" };
    setUser(newUser);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(newUser));
  };

  const signup = (name: string, email: string) => {
    const newUser: User = { name, email, provider: "email" };
    setUser(newUser);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(newUser));
  };

  const loginWithGoogle = () => {
    const newUser: User = {
      name: "Google User",
      email: "user@gmail.com",
      provider: "google",
    };
    setUser(newUser);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(newUser));
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem(STORAGE_KEY);
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        isLoggedIn: !!user,
        login,
        signup,
        loginWithGoogle,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}

