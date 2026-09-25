import { createFileRoute } from "@tanstack/react-router";
import { useCallback, useRef, useState } from "react";
import type { EditJob, GenerationJob, VideoMetadata } from "#/lib/api";
import { api } from "#/lib/api";

export const Route = createFileRoute("/")({ component: App });

type Phase = "empty" | "generating" | "ready" | "editing" | "edited";

interface Turn {
	id: number;
	kind: "generate" | "edit";
	prompt: string;
	summary?: string;
	failed?: boolean;
}

let turnCounter = 0;

function App() {
	const [phase, setPhase] = useState<Phase>("empty");
	const [prompt, setPrompt] = useState("");
	const [video, setVideo] = useState<VideoMetadata | null>(null);
	const [editJob, setEditJob] = useState<EditJob | null>(null);
	const [progress, setProgress] = useState<{
		pct: number;
		message: string;
	} | null>(null);
	const [history, setHistory] = useState<Turn[]>([]);
	const [error, setError] = useState<string | null>(null);
	// Off by default: a new prompt regenerates the whole video. Turning it on
	// switches to a localized edit that leaves the rest of the frame untouched.
	const [editMode, setEditMode] = useState(false);
	const inputRef = useRef<HTMLTextAreaElement>(null);

	const busy = phase === "generating" || phase === "editing";

	const pushTurn = useCallback((turn: Omit<Turn, "id">) => {
		turnCounter += 1;
		setHistory((prev) => [...prev, { ...turn, id: turnCounter }]);
	}, []);

	const handleGenerate = useCallback(
		async (text: string) => {
			setError(null);
			setPhase("generating");
			setProgress({ pct: 0, message: "Queued" });
			setEditJob(null);

			try {
				// Frame count, size, steps and fps come from the backend's
				// configured generator recipe (Wan2.2: 720p@24fps); hardcoding
				// the old AnimateDiff values here produced blotchy 480p renders.
				let job: GenerationJob = await api.generate({ prompt: text });

				while (job.status === "pending" || job.status === "processing") {
					await new Promise((r) => setTimeout(r, 1500));
					job = await api.getGeneration(job.id);
					setProgress({
						pct: Math.round(job.progress * 100),
						message: job.message ?? "Generating",
					});
				}

				if (job.status === "failed" || !job.result) {
					throw new Error(job.error ?? "Generation failed");
				}

				const meta = await api.getVideo(job.result.video_id);
				setVideo(meta);
				setPhase("ready");
				setProgress(null);
				pushTurn({
					kind: "generate",
					prompt: text,
					summary: `${meta.width}×${meta.height} · ${meta.frame_count} frames · ${meta.fps.toFixed(0)} fps`,
				});
			} catch (err) {
				const message =
					err instanceof Error ? err.message : "Generation failed";
				setError(message);
				setPhase("empty");
				setProgress(null);
				pushTurn({ kind: "generate", prompt: text, failed: true });
			}
		},
		[pushTurn],
	);

	const handleEdit = useCallback(
		async (text: string) => {
			if (!video) return;
			setError(null);
			setPhase("editing");
			setProgress({ pct: 0, message: "Queued" });

			try {
				let job = await api.createJob({
					video_id: video.id,
					edit_type: "inpaint",
					time_range: { start_ms: 0, end_ms: video.duration_ms },
					mask: {
						mask_type: "segmentation",
						segmentation_prompt: text,
						dilation_px: 4,
						feather_px: 2,
					},
					prompt: text,
					denoise_strength: 0.8,
					context_frames: 8,
				});

				while (job.status === "pending" || job.status === "processing") {
					await new Promise((r) => setTimeout(r, 1500));
					job = await api.getJob(job.id);
					setProgress({
						pct: Math.round(job.progress * 100),
						message: job.message ?? "Editing",
					});
				}

				if (job.status === "failed") {
					throw new Error(job.error ?? "Edit failed");
				}

				setEditJob(job);
				setPhase("edited");
				setProgress(null);

				const stats = job.result?.mask_stats;
				const changed =
					job.result?.preservation_report.changed_pixels_outside_mask ?? 0;
				pushTurn({
					kind: "edit",
					prompt: text,
					summary: `${stats?.method ?? "edited"} · ${changed} pixels changed outside the region`,
				});
			} catch (err) {
				const message = err instanceof Error ? err.message : "Edit failed";
				setError(message);
				setPhase(video ? "ready" : "empty");
				setProgress(null);
				pushTurn({ kind: "edit", prompt: text, failed: true });
			}
		},
		[video, pushTurn],
	);

	const submit = useCallback(() => {
		const text = prompt.trim();
		if (!text || busy) return;
		setPrompt("");
		// Edit mode = change only the named part. Otherwise (default) make a fresh
		// video from the new description.
		if (video && editMode) {
			handleEdit(text);
		} else {
			handleGenerate(text);
		}
	}, [prompt, busy, video, editMode, handleEdit, handleGenerate]);

	const handleKeyDown = useCallback(
		(e: React.KeyboardEvent) => {
			if (e.key === "Enter" && !e.shiftKey) {
				e.preventDefault();
				submit();
			}
		},
		[submit],
	);

	const reset = useCallback(() => {
		setVideo(null);
		setEditJob(null);
		setPrompt("");
		setError(null);
		setHistory([]);
		setPhase("empty");
		setProgress(null);
		setEditMode(false);
		inputRef.current?.focus();
	}, []);

	const showEdited = phase === "edited" && editJob?.status === "completed";
	const report = editJob?.result?.preservation_report;
	const stats = editJob?.result?.mask_stats;

	return (
		<div className="min-h-screen flex flex-col">
			<header className="px-6 py-4 border-b border-default flex items-center justify-between">
				<div>
					<h1 className="text-lg font-semibold">Preserve</h1>
					<p className="text-xs text-muted">
						Generate a video, then change one part of it
					</p>
				</div>
				{video && (
					<button
						type="button"
						onClick={reset}
						className="text-sm text-muted hover:text-default transition-colors"
					>
						New video
					</button>
				)}
			</header>

			<main className="flex-1 flex flex-col items-center justify-center p-6 w-full">
				{!video && !busy && (
					<div className="text-center max-w-lg space-y-3">
						<h2 className="text-2xl font-medium">What should I create?</h2>
						<p className="text-muted text-sm">
							Describe a video. Once it exists, ask for a change and only that
							part of the picture will be touched.
						</p>
						<div className="flex flex-wrap gap-2 justify-center pt-2">
							{[
								"a red sports car driving on an empty highway",
								"a person walking a dog through a park",
								"a yellow bus on a city street",
							].map((example) => (
								<button
									key={example}
									type="button"
									onClick={() => setPrompt(example)}
									className="px-3 py-1.5 text-xs bg-pill border border-default rounded-lg text-muted hover:text-default transition-colors"
								>
									{example}
								</button>
							))}
						</div>
					</div>
				)}

				{busy && (
					<div className="w-full max-w-md space-y-4 text-center">
						<Spinner />
						<div className="space-y-2">
							<p className="text-sm">
								{phase === "generating" ? "Generating video" : "Applying edit"}
							</p>
							<div className="h-1.5 bg-pill rounded-full overflow-hidden">
								<div
									className="h-full bg-action transition-all duration-500"
									style={{ width: `${progress?.pct ?? 0}%` }}
								/>
							</div>
							<p className="text-xs text-muted">
								{progress?.message} · {progress?.pct ?? 0}%
							</p>
						</div>
					</div>
				)}

				{video && !busy && (
					<div className="w-full max-w-5xl space-y-4">
						{showEdited ? (
							<div className="grid grid-cols-2 gap-4">
								<Player label="Before" src={api.getVideoStreamUrl(video.id)} />
								<Player label="Edited" src={api.getResultUrl(editJob.id)} />
							</div>
						) : (
							<div className="space-y-2">
								<div className="flex items-center justify-between">
									<p className="text-xs text-muted">Your video</p>
									<button
										type="button"
										onClick={() => setEditMode((v) => !v)}
										className={`px-3 py-1 rounded-lg text-xs font-medium border transition-colors ${
											editMode
												? "bg-action text-[#171615] border-transparent"
												: "bg-pill border-default text-muted hover:text-default"
										}`}
									>
										{editMode ? "✓ Editing one part" : "Edit one part"}
									</button>
								</div>
								<div className="bg-surface border border-default rounded-xl overflow-hidden">
									<video
										src={api.getVideoStreamUrl(video.id)}
										controls
										loop
										className="w-full"
									>
										<track kind="captions" />
									</video>
								</div>
							</div>
						)}

						{showEdited && report && (
							<div className="flex items-center justify-between bg-surface border border-default rounded-xl p-4">
								<div className="space-y-1">
									<p className="font-medium text-sm">
										{report.passed
											? "Only the requested region changed"
											: "Preservation check failed"}
									</p>
									<p className="text-xs text-muted">
										{report.changed_pixels_outside_mask} pixels changed outside
										the edit region
										{stats?.method ? ` · ${stats.method}` : ""}
										{stats?.generative === false
											? " · no generative model"
											: ""}
									</p>
									{stats?.edit && (
										<p className="text-xs text-muted">{stats.edit}</p>
									)}
								</div>
								<a
									href={api.getResultUrl(editJob.id)}
									download
									className="px-4 py-2 bg-action text-[#171615] rounded-lg font-medium text-sm"
								>
									Download
								</a>
							</div>
						)}

						<div className="flex gap-2 text-xs text-muted">
							<span>
								{video.width}×{video.height}
							</span>
							<span>·</span>
							<span>{video.fps.toFixed(0)} fps</span>
							<span>·</span>
							<span>{video.frame_count} frames</span>
							<span>·</span>
							<span title={video.file_hash}>
								sha256 {video.file_hash.slice(0, 12)}
							</span>
						</div>
					</div>
				)}

				{history.length > 0 && (
					<div className="w-full max-w-2xl mt-6 space-y-1.5">
						{history.map((turn) => (
							<div key={turn.id} className="flex gap-2 text-xs">
								<span className="text-muted shrink-0">
									{turn.kind === "generate" ? "created" : "edited"}
								</span>
								<span className={turn.failed ? "text-red-400" : ""}>
									{turn.prompt}
								</span>
								{turn.summary && (
									<span className="text-muted ml-auto shrink-0">
										{turn.summary}
									</span>
								)}
							</div>
						))}
					</div>
				)}

				{error && (
					<div className="mt-4 px-4 py-2 bg-red-900/30 border border-red-800/50 rounded-lg text-red-300 text-sm max-w-2xl">
						{error}
					</div>
				)}
			</main>

			<footer className="p-4">
				<div className="max-w-2xl mx-auto">
					<div className="bg-surface border border-default rounded-xl p-3">
						<textarea
							ref={inputRef}
							value={prompt}
							onChange={(e) => setPrompt(e.target.value)}
							onKeyDown={handleKeyDown}
							placeholder={
								!video
									? "Describe the video to generate…"
									: editMode
										? "Change one part… e.g. 'make the car blue' or 'remove the car'"
										: "Describe a new video…"
							}
							disabled={busy}
							rows={1}
							className="w-full bg-transparent resize-none focus:outline-none disabled:opacity-50"
						/>
						<div className="flex items-center justify-between mt-2">
							<p className="text-xs text-muted">
								{!video
									? "Press Enter to generate"
									: editMode
										? "Only the part you name changes — the rest stays identical"
										: "Enter makes a new video · turn on Edit to change one part"}
							</p>
							<button
								type="button"
								onClick={submit}
								disabled={!prompt.trim() || busy}
								className="px-4 py-1.5 bg-action text-[#171615] rounded-lg font-medium text-sm disabled:opacity-50 disabled:cursor-not-allowed hover:opacity-90 transition-opacity"
							>
								{busy
									? "Working…"
									: video && editMode
										? "Apply edit"
										: video
											? "Regenerate"
											: "Generate"}
							</button>
						</div>
					</div>
				</div>
			</footer>
		</div>
	);
}

function Player({ label, src }: { label: string; src: string }) {
	return (
		<div className="space-y-2">
			<p className="text-xs text-muted text-center">{label}</p>
			<div className="bg-surface border border-default rounded-xl overflow-hidden">
				<video src={src} controls loop className="w-full">
					<track kind="captions" />
				</video>
			</div>
		</div>
	);
}

function Spinner() {
	return (
		<svg
			className="animate-spin h-8 w-8 mx-auto text-muted"
			fill="none"
			viewBox="0 0 24 24"
			aria-label="Loading"
		>
			<title>Loading</title>
			<circle
				className="opacity-25"
				cx="12"
				cy="12"
				r="10"
				stroke="currentColor"
				strokeWidth="4"
			/>
			<path
				className="opacity-75"
				fill="currentColor"
				d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
			/>
		</svg>
	);
}
