import React, { useState } from 'react';

export function GooeyInput({ placeholder = "Search...", value, onChange }) {
  const [isFocused, setIsFocused] = useState(false);

  return (
    <div className="relative flex items-center w-full max-w-md">
      <div 
        className={`absolute inset-0 rounded-full bg-gradient-to-r from-purple-600 to-indigo-600 transition-all duration-300 blur-xl ${isFocused ? 'opacity-70 scale-105' : 'opacity-0 scale-90'}`}
        style={{ zIndex: 0 }}
      ></div>
      <div className="relative w-full z-10">
        <input 
          type="text"
          placeholder={placeholder}
          value={value}
          onChange={onChange}
          onFocus={() => setIsFocused(true)}
          onBlur={() => setIsFocused(false)}
          className="w-full h-12 px-6 py-3 bg-[#0a0018] bg-opacity-80 backdrop-blur-md rounded-full border border-purple-500/30 text-white placeholder-gray-400 focus:outline-none focus:border-purple-500/70 focus:ring-2 focus:ring-purple-500/50 transition-all duration-300 font-light tracking-wide text-sm shadow-[0_0_15px_rgba(132,0,255,0.1)]"
        />
        <div className="absolute right-4 top-1/2 -translate-y-1/2 text-purple-400">
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="11" cy="11" r="8"></circle>
            <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
          </svg>
        </div>
      </div>
    </div>
  );
}
