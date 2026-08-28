import { LogOut } from 'lucide-react';

export default function Navbar({ 
  user, 
  onOpenAuth, 
  onSignOut,
  onNavigateHome
}) {
  const getInitials = (name) => {
    if (!name) return 'U';
    return name
      .split(' ')
      .map(n => n[0])
      .join('')
      .toUpperCase()
      .slice(0, 2);
  };

  return (
    <header className="fixed top-0 left-0 right-0 z-40 h-20 transition-all duration-300 bg-gradient-to-b from-slate-950/80 via-slate-950/40 to-transparent" style={{ WebkitBackdropFilter: 'blur(12px)', backdropFilter: 'blur(12px)' }}>
      <div className="max-w-7xl mx-auto h-full px-6 md:px-12 flex items-center justify-between">
        
        {/* Left: Minimal text logo */}
        <button 
          onClick={onNavigateHome}
          className="text-left group cursor-pointer focus:outline-none"
        >
          <span className="font-display font-medium text-lg tracking-tight text-slate-100 group-hover:text-white transition-colors duration-300">
            TerraMesh
          </span>
        </button>

        {/* Right: Authentication or User Profile */}
        <div className="flex items-center gap-4">
          {!user ? (
            <>
              <button
                onClick={onOpenAuth}
                className="text-sm font-medium text-slate-400 hover:text-slate-200 transition-colors duration-300 px-3 py-2 cursor-pointer focus:outline-none"
              >
                Login
              </button>
              
              <button
                onClick={onOpenAuth}
                className="text-sm font-medium px-4 py-2 rounded-full bg-slate-100 text-slate-950 hover:bg-white transition-all duration-300 shadow-sm cursor-pointer focus:outline-none"
              >
                Try Prototype
              </button>
            </>
          ) : (
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2.5 px-3 py-1.5 rounded-full bg-slate-900/60 border border-slate-800/80" style={{ WebkitBackdropFilter: 'blur(12px)', backdropFilter: 'blur(12px)' }}>
                <div className="w-6 h-6 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center text-slate-200 text-xs font-medium">
                  {getInitials(user.name)}
                </div>
                <span className="text-sm font-medium text-slate-200">
                  {user.name}
                </span>
              </div>

              <button
                onClick={onSignOut}
                title="Sign out / Return home"
                className="p-2 rounded-full text-slate-400 hover:text-slate-200 hover:bg-slate-900/60 transition-colors duration-300 cursor-pointer focus:outline-none"
              >
                <LogOut className="w-4 h-4" />
              </button>
            </div>
          )}
        </div>

      </div>
    </header>
  );
}
