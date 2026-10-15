import "react";

declare module "react" {
  interface InputHTMLAttributes<T> extends HTMLAttributes<T> {
    /** Non-standard but supported by every major browser; lets a file input pick a folder. */
    webkitdirectory?: string;
  }
}
