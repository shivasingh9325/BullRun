import React, { useEffect, useState, useCallback } from 'react';
import Prism from './components/Prism';
import PillNav from './components/PillNav';
import MagicBento from './components/MagicBento';
import { GooeyInput } from './components/GooeyInput';
import { Activity, BarChart2, CheckCircle, AlertTriangle, Zap, Server, Shield, Target } from 'lucide-react';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api/v1';

function App() {
  const [activeTab, setActiveTab] = useState('/');
  const [searchValue, setSearchValue] = useState('');
  
  // Data States
  const [health, setHealth] = useState({ status: 'connecting...', uptime_seconds: 0, errors_count: 0 });
  const [portfolio, setPortfolio] = useState({ fiat_balance: 0, total_value: 0, unrealized_pnl: 0, holdings: {} });
  const [signals, setSignals] = useState([]);
  const [isPredicting, setIsPredicting] = useState(false);
  const [insightData, setInsightData] = useState({
    topSignal: null,
    highestConfidence: 0
  });
  
  const fetchHealth = useCallback(async () => {
    try {
      const res = await axios.get(`${API_BASE_URL}/health`);
      setHealth(res.data);
    } catch (err) {
      setHealth(prev => ({ ...prev, status: 'Error connecting' }));
      console.error(err);
    }
  }, []);

  const fetchPortfolio = useCallback(async () => {
    try {
      const res = await axios.get(`${API_BASE_URL}/portfolio`);
      setPortfolio(res.data);
    } catch (err) {
      console.error(err);
    }
  }, []);

  const handlePredict = async () => {
    setIsPredicting(true);
    try {
      const payload = searchValue 
        ? { symbol: searchValue } 
        : { symbol: 'RELIANCE.NS' };
      
      const res = await axios.post(`${API_BASE_URL}/predict`, payload);
      
      // Update signals list with new decisions
      setSignals(res.data.decisions || []);
      
      // Extract insights from new decisions
      if (res.data.decisions?.length > 0) {
         const top = [...res.data.decisions].sort((a,b) => b.confidence - a.confidence)[0];
         setInsightData({ topSignal: top, highestConfidence: top.confidence * 100 });
      }
      
      // Refetch portfolio to show updated values
      fetchPortfolio();
    } catch (err) {
      console.error("Prediction failed:", err);
      // Optional: Add a toast/error pop-up handler here
    } finally {
      setIsPredicting(false);
    }
  };

  useEffect(() => {
    fetchHealth();
    fetchPortfolio();
    
    // Initial fetch of trades history
    axios.get(`${API_BASE_URL}/portfolio/trades?limit=5`)
      .then(res => {
         if(res.data && res.data.length > 0) {
            // Map past trades to active 'signals' view for dashboard context
            const pastSignals = res.data.map(t => ({
               symbol: t.ticker,
               signal: t.action,
               confidence: (t.confidence || 0) * 100,
               allocation: (t.quantity || 0),
               time: new Date(t.timestamp).toLocaleTimeString(),
               raw: t
            }));
            setSignals(pastSignals);
         }
      })
      .catch(console.error);

    const interval = setInterval(() => {
      fetchHealth();
      fetchPortfolio();
    }, 15000); // 15s refresh
    
    return () => clearInterval(interval);
  }, [fetchHealth, fetchPortfolio]);

  const navItems = [
    { label: 'Dashboard', href: '/' },
    { label: 'Signals', href: '/#signals' },
    { label: 'Portfolio', href: '/#portfolio' },
    { label: 'System', href: '/#system' }
  ];

  return (
    <div className="min-h-screen bg-[#060010] text-white relative overflow-x-hidden font-sans">
      {/* 3D Background Effect */}
      <div className="fixed inset-0 z-0 opacity-40 pointer-events-none">
        <Prism 
          animationType="rotate"
          timeScale={0.2}
          height={3.5}
          baseWidth={5.5}
          scale={3.6}
          glow={1.2}
          noise={0.1}
          hueShift={-0.1} // Shift to slightly cooler purple/blue
          colorFrequency={1.2}
        />
      </div>

      {/* Navigation */}
      <header className="relative z-50 flex justify-center w-full pt-6 px-4">
        <PillNav
          logo={null}
          items={navItems}
          activeHref={activeTab}
          baseColor="#8400ff"
          pillColor="rgba(10, 0, 24, 0.8)"
          hoveredPillTextColor="#ffffff"
          pillTextColor="#d8b4fe"
        />
      </header>

      {/* Main Content */}
      <main className="relative z-10 container mx-auto px-6 pt-24 pb-20 flex flex-col items-center">
        
        {/* Header Section */}
        <div className="w-full max-w-5xl mb-12 flex flex-col md:flex-row justify-between items-center gap-6">
          <div>
            <h1 className="text-4xl md:text-5xl font-light tracking-tight mb-2">
              BullRun <span className="font-semibold text-purple-400">AI</span>
            </h1>
            <p className="text-gray-400 text-sm md:text-base">Advanced AI Trading & Portfolio Management</p>
          </div>
          
          <div className="w-full md:w-auto flex gap-4 items-center">
            <GooeyInput 
              placeholder="Target symbol..." 
              value={searchValue}
              onChange={(e) => setSearchValue(e.target.value)}
            />
            <button 
              onClick={handlePredict}
              disabled={isPredicting}
              className={`px-5 py-3 whitespace-nowrap rounded-xl text-white text-sm font-medium transition-all border 
                ${isPredicting ? 'bg-gray-800 border-gray-600 opacity-70 cursor-not-allowed' : 'bg-purple-600 hover:bg-purple-500 border-purple-500'}`}
            >
              {isPredicting ? 'Running Pipeline...' : 'Run Pipeline'}
            </button>
          </div>
        </div>

        {/* Action Prediction (Insights) */}
        <section className="w-full max-w-5xl mb-10 bg-[#0a0018]/60 backdrop-blur-md rounded-2xl border border-purple-900/30 p-6 flex flex-col md:flex-row shadow-[0_4px_30px_rgba(132,0,255,0.05)]">
           <div className="flex-1 pr-6 border-b md:border-b-0 md:border-r border-purple-900/30 mb-6 md:mb-0">
             <div className="flex items-center gap-3 mb-4 text-purple-300">
               <Target size={20} />
               <h2 className="text-lg font-medium">Metamodel Consensus</h2>
             </div>
             <p className="text-gray-300 font-light leading-relaxed mb-4 text-sm">
               {insightData.topSignal 
                 ? `The latest inference highlights a ${insightData.topSignal.signal} signal for ${insightData.topSignal.symbol} with ${insightData.highestConfidence.toFixed(1)}% confidence. Sentiment is evaluated and RL constraints are mapped.`
                 : "The reinforcement learning agent aggregates structural data. Awaiting inference..."}
             </p>
           </div>
           
           <div className="flex-1 md:pl-6 grid grid-cols-2 gap-4">
              <div className="bg-[#0f0022] rounded-xl p-4 border border-purple-900/40">
                 <div className="text-xs text-gray-400 mb-1 flex justify-between">
                   <span>Latest Signal</span>
                   <Activity size={14} className="text-green-400"/>
                 </div>
                 <div className="text-xl font-bold text-white">
                   {insightData.topSignal ? insightData.topSignal.signal : 'WAITING'}
                 </div>
                 <div className="text-xs text-green-400 mt-1">
                   {insightData.topSignal ? `${insightData.topSignal.symbol} (${insightData.highestConfidence.toFixed(1)}%)` : '--'}
                 </div>
              </div>
              <div className="bg-[#0f0022] rounded-xl p-4 border border-purple-900/40">
                 <div className="text-xs text-gray-400 mb-1 flex justify-between">
                   <span>Risk Level</span>
                   <Shield size={14} className="text-yellow-400"/>
                 </div>
                 <div className="text-xl font-bold text-white">Controlled</div>
                 <div className="text-xs text-yellow-400 mt-1">RL Drawdown Guard</div>
              </div>
              <div className="bg-[#0f0022] rounded-xl p-4 border border-purple-900/40 col-span-2 flex justify-between items-center">
                 <div>
                   <div className="text-xs text-gray-400 mb-1">Portfolio Equity</div>
                   <div className="text-2xl font-bold text-white">${portfolio.total_value.toLocaleString()}</div>
                 </div>
                 <div className="text-right">
                   <div className="text-xs text-gray-400 mb-1">Unrealized PnL</div>
                   <div className={`text-lg font-bold ${portfolio.unrealized_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                     {portfolio.unrealized_pnl >= 0 ? '+' : ''}${portfolio.unrealized_pnl.toLocaleString()}
                   </div>
                 </div>
              </div>
           </div>
        </section>

        {/* Interactive Bento Dashboard */}
        <div id="dashboard" className="w-full flex justify-center">
          <MagicBento 
            enableStars={true}
            enableSpotlight={true}
            enableBorderGlow={true}
            particleCount={16}
            glowColor="132, 0, 255"
            clickEffect={true}
            enableTilt={false}
          />
        </div>

        {/* Detailed Signals & System section */}
        <div className="w-full max-w-5xl mt-12 grid grid-cols-1 lg:grid-cols-3 gap-6" id="signals">
          
          {/* Active Signals List */}
          <div className="lg:col-span-2 bg-[#0a0018]/60 backdrop-blur-md rounded-2xl border border-purple-900/30 p-6">
            <div className="flex items-center gap-3 mb-6">
              <Zap size={20} className="text-purple-400" />
              <h2 className="text-lg font-medium text-white">Recent Decisions & Trades</h2>
            </div>
            
            <div className="space-y-4">
              {signals.length === 0 ? (
                <div className="text-gray-500 text-center py-6">No recent signals found. Run pipeline.</div>
              ) : signals.map((sig, i) => (
                <div key={i} className="bg-[#0f0022] p-4 rounded-xl border border-purple-900/20 flex flex-wrap justify-between items-center gap-4 hover:border-purple-500/30 transition-colors">
                  <div className="flex items-center gap-4">
                    <div className={`w-10 h-10 rounded-full flex items-center justify-center font-bold text-xs
                      ${sig.signal === 'BUY' ? 'bg-green-500/20 text-green-400' : sig.signal === 'SELL' ? 'bg-red-500/20 text-red-400' : 'bg-gray-500/20 text-gray-300'}`}>
                      {sig.signal}
                    </div>
                    <div>
                      <div className="font-semibold text-white">{sig.symbol}</div>
                      <div className="text-xs text-gray-400">{sig.time || new Date().toLocaleTimeString()}</div>
                    </div>
                  </div>
                  
                  <div className="flex gap-6 items-center">
                    <div className="text-center min-w-[60px]">
                       <div className="text-xs text-gray-400">Confidence</div>
                       <div className="text-sm font-medium text-purple-300">
                         {sig.confidence ? (sig.confidence > 1 ? sig.confidence : (sig.confidence * 100)).toFixed(1) : 0}%
                       </div>
                    </div>
                    <div className="text-center min-w-[60px]">
                       <div className="text-xs text-gray-400">Alloc Qty</div>
                       <div className="text-sm font-medium text-gray-200">
                         {sig.allocation != null ? parseFloat(sig.allocation).toFixed(2) : '0'}
                       </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
          
          {/* System Health */}
          <div className="bg-[#0a0018]/60 backdrop-blur-md rounded-2xl border border-purple-900/30 p-6 flex flex-col" id="system">
            <div className="flex items-center gap-3 mb-6">
              <Server size={20} className="text-purple-400" />
              <h2 className="text-lg font-medium text-white">System Status</h2>
            </div>

            <div className="flex-1 space-y-6">
               <div className="flex justify-between items-center">
                 <span className="text-sm text-gray-400">Backend API</span>
                 <div className="flex items-center gap-2">
                   <CheckCircle size={14} className={health.status === 'Operational' ? "text-green-400" : "text-yellow-400"} />
                   <span className={`text-sm font-medium ${health.status === 'Operational' ? "text-green-400" : "text-yellow-400"}`}>
                     {health.status}
                   </span>
                 </div>
               </div>
               
               <div className="flex justify-between items-center">
                 <span className="text-sm text-gray-400">Database</span>
                 <div className="flex items-center gap-2">
                   <Server size={14} className="text-green-400" />
                   <span className="text-sm text-green-400 font-medium">Connected</span>
                 </div>
               </div>

               <div className="flex justify-between items-center">
                 <span className="text-sm text-gray-400">Model Engine</span>
                 <div className="flex items-center gap-2">
                   <CheckCircle size={14} className="text-green-400" />
                   <span className="text-sm text-green-400 font-medium">Ready</span>
                 </div>
               </div>
               
               <div className="mt-6 pt-6 border-t border-purple-900/30">
                 <div className="flex justify-between text-xs text-gray-400 mb-2">
                   <span>Uptime</span>
                   <span>{health.uptime_seconds}s</span>
                 </div>
                 <div className="w-full bg-gray-900 rounded-full h-1.5 overflow-hidden">
                   <div className="bg-gradient-to-r from-purple-500 to-indigo-500 h-1.5 rounded-full" style={{ width: '100%' }}></div>
                 </div>
               </div>
            </div>
          </div>
        </div>

      </main>
    </div>
  );
}

export default App;

