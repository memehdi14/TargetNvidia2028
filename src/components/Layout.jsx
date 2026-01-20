import { useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { LayoutDashboard, ListTodo, Menu, X, Trophy } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const Layout = ({ children }) => {
    const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
    const location = useLocation();

    const navItems = [
        { to: "/", icon: <LayoutDashboard size={22} />, label: "Mission Control" },
        { to: "/plan", icon: <ListTodo size={22} />, label: "The Plan" },
        { to: "/achievements", icon: <Trophy size={22} />, label: "Achievements" },
    ];

    return (
        <div className="min-h-screen flex flex-col font-sans selection:bg-primary/30 selection:text-white">
            {/* Top Navbar */}
            <nav className="fixed top-0 left-0 right-0 z-50 bg-black/50 backdrop-blur-md border-b border-white/5 h-16">
                <div className="container h-full flex items-center justify-between">

                    {/* Logo Area */}
                    <div className="flex items-center gap-3">
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            className="w-8 h-8 flex items-center justify-center text-primary drop-shadow-[0_0_8px_rgba(118,185,0,0.8)]"
                        >
                            <svg viewBox="0 0 24 24" fill="currentColor" className="w-full h-full">
                                <path d="M21.6 15.6l-1.6-9.1c-.2-1.1-1.1-1.9-2.2-1.9H6.2c-1.1 0-2 .8-2.2 1.9l-1.6 9.1c-1 5.9 3.5 11 9.4 11s10.4-5.1 9.4-11zM12 21.2c-2.3 0-4.4-.9-5.9-2.4l.6-1.5c1.4 1.2 3.2 1.9 5.3 1.9 2.1 0 3.9-.7 5.3-1.9l.6 1.5c-1.5 1.5-3.6 2.4-5.9 2.4zm0-5c-2.3 0-4.2-1.9-4.2-4.2s1.9-4.2 4.2-4.2 4.2 1.9 4.2 4.2-1.9 4.2-4.2 4.2zm6.7-7.6l.8 4.4H4.5l.8-4.4c.1-.5.5-.9 1-.9h11.4c.5 0 .9.4 1 .9z" />
                                <path d="M11.5 8c-.6 0-1.1.2-1.5.5L7.9 5.8c1-.7 2.3-1.1 3.6-1.1 3.5 0 6.4 2.6 6.9 6h-1.6c-.5-2.5-2.7-4.4-5.3-4.4z" />
                                <path d="M24 10.6h-1.6c-.3-2.3-1.6-4.3-3.4-5.7l1-1.6c2.4 1.7 4 4.5 4 7.3z" />
                            </svg>
                        </motion.div>
                        <div className="flex flex-col leading-none">
                            <span className="font-bold text-lg tracking-wide text-white">TARGET <span className="text-primary text-glow drop-shadow-[0_0_5px_rgba(118,185,0,0.6)]">NVIDIA</span></span>
                            <span className="text-[10px] text-gray-400 font-mono tracking-[0.2em]">MAY 2028</span>
                        </div>
                    </div>

                    {/* Desktop Nav */}
                    <div className="hidden md:flex items-center gap-2">
                        {navItems.map((item) => (
                            <NavLink
                                key={item.to}
                                to={item.to}
                                className={({ isActive }) => `nav-item ${isActive ? 'active' : 'text-gray-400'}`}
                            >
                                {item.icon}
                                <span className="text-sm font-medium">{item.label}</span>
                            </NavLink>
                        ))}
                    </div>

                    {/* Mobile Menu Button */}
                    <button
                        className="md:hidden text-white p-2 active:scale-95 transition-transform"
                        onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
                    >
                        {isMobileMenuOpen ? <X size={26} /> : <Menu size={26} />}
                    </button>
                </div>
            </nav>

            {/* Mobile Nav Overlay */}
            <AnimatePresence>
                {isMobileMenuOpen && (
                    <motion.div
                        initial={{ opacity: 0, y: -20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                        className="fixed inset-0 z-40 bg-black/95 pt-20 px-4 md:hidden backdrop-blur-xl"
                    >
                        <div className="flex flex-col gap-3">
                            {navItems.map((item) => (
                                <NavLink
                                    key={item.to}
                                    to={item.to}
                                    onClick={() => setIsMobileMenuOpen(false)}
                                    className={({ isActive }) =>
                                        `flex items-center gap-4 p-5 rounded-2xl border ${isActive
                                            ? "bg-primary/10 border-primary/50 text-white shadow-[0_0_15px_-5px_var(--primary-glow)]"
                                            : "bg-white/5 border-white/10 text-gray-400"
                                        }`
                                    }
                                >
                                    <div className={`${isActive ? "text-primary" : ""}`}>{item.icon}</div>
                                    <span className="text-lg font-bold">{item.label}</span>
                                </NavLink>
                            ))}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Main Content */}
            <main className="container pt-24 pb-24 flex-grow">
                <motion.div
                    key={location.pathname}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.4 }}
                >
                    {children}
                </motion.div>
            </main>
        </div>
    );
};

export default Layout;
