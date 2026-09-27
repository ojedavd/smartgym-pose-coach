import type { Metadata } from 'next';
import { Geist, Geist_Mono } from 'next/font/google';
import './globals.css';

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
});

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
});

export const metadata: Metadata = {
  metadataBase: new URL(process.env.NEXT_PUBLIC_BASE_URL || 'http://localhost:3000'),
  title: 'Forma · Motion Coach',
  description: 'Entrenamiento guiado con análisis de postura y feedback en tiempo real.',
  manifest: '/manifest.webmanifest',
  applicationName: 'Forma',
  appleWebApp: {
    capable: true,
    statusBarStyle: 'black-translucent',
    title: 'Forma',
  },
  icons: {
    icon: '/icon.svg',
    apple: '/icon.svg',
  },
  openGraph: {
    type: 'website',
    title: 'Forma · Motion Coach',
    description: 'Entrená mejor. Movimiento por movimiento.',
    images: [{ url: '/og.png', width: 1200, height: 630, alt: 'Forma Motion Coach' }],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Forma · Motion Coach',
    description: 'Entrená mejor. Movimiento por movimiento.',
    images: ['/og.png'],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="es" className="dark">
      <head>
        <meta name="theme-color" content="#0b1117" />
      </head>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
