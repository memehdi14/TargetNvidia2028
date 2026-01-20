import { useState } from 'react';
import { useTracker } from '../context/TrackerContext';
import { ChevronDown, ChevronRight, Check, Calendar, BookOpen, HelpCircle, ExternalLink } from 'lucide-react';
import ProgressBar from './ProgressBar';

const PlanView = () => {
    const { planData, completedTasks, toggleTask } = useTracker();

    // Default expand the first phase
    const [expandedPhases, setExpandedPhases] = useState([planData.phases[0].id]);
    const [expandedMonths, setExpandedMonths] = useState([planData.phases[0].months[0].id]);

    const togglePhase = (id) => {
        setExpandedPhases(prev =>
            prev.includes(id) ? prev.filter(p => p !== id) : [...prev, id]
        );
    };

    const toggleMonth = (id) => {
        setExpandedMonths(prev =>
            prev.includes(id) ? prev.filter(m => m !== id) : [...prev, id]
        );
    };

    const isTaskComplete = (id) => completedTasks.includes(id);

    // Depth level color mapping
    const getDepthColor = (depth) => {
        if (!depth) return 'bg-gray-500/20 text-gray-400';
        if (depth.includes('🔴') || depth.includes('Expert')) return 'bg-red-500/20 text-red-400 border-red-500/30';
        if (depth.includes('🟡') || depth.includes('Working')) return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
        if (depth.includes('🟢') || depth.includes('Awareness')) return 'bg-green-500/20 text-green-400 border-green-500/30';
        if (depth.includes('Certificate') || depth.includes('Project')) return 'bg-purple-500/20 text-purple-400 border-purple-500/30';
        return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
    };

    return (
        <div className="space-y-6 pb-20">
            <div className="flex flex-col gap-1">
                <h1 className="text-3xl font-bold">The Plan</h1>
                <p className="text-muted">18 Months to NVIDIA • 1hr/day = 7hr/week</p>
            </div>

            {/* Depth Legend */}
            <div className="glass-panel p-4">
                <h3 className="text-sm font-bold text-muted mb-3">DEPTH LEVELS</h3>
                <div className="flex flex-wrap gap-2">
                    <span className="px-2 py-1 text-xs rounded border bg-red-500/20 text-red-400 border-red-500/30">🔴 Expert - Can implement & optimize</span>
                    <span className="px-2 py-1 text-xs rounded border bg-yellow-500/20 text-yellow-400 border-yellow-500/30">🟡 Working - Can implement from scratch</span>
                    <span className="px-2 py-1 text-xs rounded border bg-green-500/20 text-green-400 border-green-500/30">🟢 Awareness - Can explain concept</span>
                    <span className="px-2 py-1 text-xs rounded border bg-purple-500/20 text-purple-400 border-purple-500/30">📜 Cert/Project</span>
                </div>
            </div>

            <div className="space-y-4">
                {planData.phases.map(phase => (
                    <div key={phase.id} className="glass-panel overflow-hidden">
                        {/* Phase Header */}
                        <div
                            onClick={() => togglePhase(phase.id)}
                            className="p-4 flex items-center justify-between cursor-pointer hover:bg-white/5 transition-colors"
                        >
                            <div className="flex flex-col">
                                <h2 className="text-lg font-bold text-white flex items-center gap-2">
                                    {phase.title}
                                </h2>
                                <span className="text-sm text-primary">{phase.subtitle}</span>
                            </div>
                            {expandedPhases.includes(phase.id) ? <ChevronDown /> : <ChevronRight />}
                        </div>

                        {/* Phase Content (Months) */}
                        {expandedPhases.includes(phase.id) && (
                            <div className="border-t border-white/10 bg-black/20">
                                {phase.months.map(month => (
                                    <div key={month.id} className="border-b border-white/5 last:border-0">
                                        {/* Month Header */}
                                        <div
                                            onClick={() => toggleMonth(month.id)}
                                            className="p-3 px-4 flex items-center justify-between cursor-pointer hover:bg-white/5"
                                        >
                                            <div className="flex items-center gap-3">
                                                <Calendar size={16} className="text-muted" />
                                                <span className="font-medium">{month.name}</span>
                                            </div>
                                            {expandedMonths.includes(month.id) ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                                        </div>

                                        {/* Weeks & Tasks */}
                                        {expandedMonths.includes(month.id) && (
                                            <div className="p-4 pt-0 space-y-4">
                                                {month.weeks.map(week => (
                                                    <div key={week.id} className="ml-2 pl-4 border-l-2 border-primary/30 mt-4">
                                                        {/* Week Header with Depth Badge */}
                                                        <div className="mb-3">
                                                            <div className="flex items-center gap-2 flex-wrap mb-1">
                                                                <h4 className="text-sm font-bold text-primary">{week.title}</h4>
                                                                {week.depth && (
                                                                    <span className={`px-2 py-0.5 text-xs rounded border ${getDepthColor(week.depth)}`}>
                                                                        {week.depth}
                                                                    </span>
                                                                )}
                                                            </div>
                                                            <p className="text-xs text-muted uppercase tracking-wider">{week.focus} • {week.hours}h</p>

                                                            {/* Reference Link */}
                                                            {week.reference && (
                                                                <div className="flex items-center gap-1 mt-2">
                                                                    <BookOpen size={12} className="text-blue-400" />
                                                                    <a
                                                                        href={week.reference.startsWith('http') ? week.reference : `https://${week.reference}`}
                                                                        target="_blank"
                                                                        rel="noopener noreferrer"
                                                                        className="text-xs text-blue-400 hover:underline truncate max-w-[300px]"
                                                                        onClick={(e) => e.stopPropagation()}
                                                                    >
                                                                        {week.reference}
                                                                    </a>
                                                                </div>
                                                            )}

                                                            {/* Depth Check Question */}
                                                            {week.depthCheck && week.depthCheck !== 'N/A - Complete course modules' && (
                                                                <div className="mt-2 p-2 rounded bg-white/5 border border-white/10">
                                                                    <div className="flex items-start gap-2">
                                                                        <HelpCircle size={14} className="text-yellow-400 mt-0.5 flex-shrink-0" />
                                                                        <p className="text-xs text-yellow-200 italic">
                                                                            Depth Check: {week.depthCheck}
                                                                        </p>
                                                                    </div>
                                                                </div>
                                                            )}
                                                        </div>

                                                        {/* Tasks */}
                                                        <div className="space-y-2">
                                                            {week.tasks.map((task, idx) => {
                                                                const taskId = `${week.id}-${idx}`;
                                                                const isLeetCode = task.toLowerCase().includes('leetcode') || task.toLowerCase().includes('solve:');
                                                                const isMilestone = task.toLowerCase().includes('milestone') || task.toLowerCase().includes('claim:');

                                                                return (
                                                                    <div
                                                                        key={taskId}
                                                                        onClick={() => toggleTask(taskId)}
                                                                        className={`
                                                                            group flex items-start gap-3 p-3 rounded-lg cursor-pointer transition-all
                                                                            ${isTaskComplete(taskId) ? 'bg-primary/10' : 'bg-surface hover:bg-surface-hover'}
                                                                            ${isMilestone ? 'border border-yellow-500/30' : ''}
                                                                        `}
                                                                    >
                                                                        <div className={`
                                                                            mt-0.5 w-5 h-5 rounded border flex items-center justify-center transition-colors flex-shrink-0
                                                                            ${isTaskComplete(taskId) ? 'bg-primary border-primary' : 'border-gray-500 group-hover:border-primary'}
                                                                        `}>
                                                                            {isTaskComplete(taskId) && <Check size={14} className="text-black" strokeWidth={3} />}
                                                                        </div>
                                                                        <span className={`text-sm ${isTaskComplete(taskId) ? 'text-gray-400 line-through' : 'text-gray-200'} ${isLeetCode ? 'font-mono' : ''} ${isMilestone ? 'text-yellow-300 font-semibold' : ''}`}>
                                                                            {task}
                                                                        </span>
                                                                    </div>
                                                                )
                                                            })}
                                                        </div>
                                                    </div>
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                ))}
            </div>
        </div>
    );
};

export default PlanView;
