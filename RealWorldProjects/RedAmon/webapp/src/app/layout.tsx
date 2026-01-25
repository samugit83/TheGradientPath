import type { Metadata } from 'next'
import '@/styles/index.css'
import { QueryProvider } from '@/providers/QueryProvider'
import { ToastProvider } from '@/components/ui'
import { AppLayout } from '@/components/layout'

export const metadata: Metadata = {
  title: 'RedAmon - Security Reconnaissance Dashboard',
  description: 'Security reconnaissance and vulnerability assessment dashboard',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        {/* Prevent flash of wrong theme */}
        <script
          dangerouslySetInnerHTML={{
            __html: `
              (function() {
                try {
                  var theme = localStorage.getItem('redamon-theme');
                  if (theme === 'dark' || theme === 'light') {
                    document.documentElement.setAttribute('data-theme', theme);
                  } else if (window.matchMedia('(prefers-color-scheme: light)').matches) {
                    document.documentElement.setAttribute('data-theme', 'light');
                  } else {
                    document.documentElement.setAttribute('data-theme', 'dark');
                  }
                } catch (e) {
                  document.documentElement.setAttribute('data-theme', 'dark');
                }
              })();
            `,
          }}
        />
      </head>
      <body>
        <QueryProvider>
          <ToastProvider>
            <AppLayout>{children}</AppLayout>
          </ToastProvider>
        </QueryProvider>
      </body>
    </html>
  )
}
