"use client"

import { useState } from "react"
import { Upload, ImageIcon, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { classifyImage, ClassificationResult } from "@/lib/api"
import Image from "next/image"

export function HeritageClassifier() {
    const [selectedImage, setSelectedImage] = useState<File | null>(null)
    const [previewUrl, setPreviewUrl] = useState<string | null>(null)
    const [result, setResult] = useState<ClassificationResult | null>(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0]
        if (file) {
            setSelectedImage(file)
            setPreviewUrl(URL.createObjectURL(file))
            setResult(null)
            setError(null)
        }
    }

    const handleClassify = async () => {
        if (!selectedImage) return

        setLoading(true)
        setError(null)
        try {
            const data = await classifyImage(selectedImage)
            setResult(data)
        } catch (err) {
            setError("Failed to classify image. Ensure the API is running.")
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="grid gap-6 md:grid-cols-2">
            {/* Upload Section */}
            <Card>
                <CardHeader>
                    <CardTitle>Upload Photo</CardTitle>
                    <CardDescription>Upload a detailed photo of a collaborative heritage site.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                    <div className="relative flex flex-col items-center justify-center rounded-lg border-2 border-dashed border-stone-300 bg-stone-50 p-6 transition-colors hover:bg-stone-100 dark:border-stone-700 dark:bg-stone-900">
                        {previewUrl ? (
                            <div className="relative aspect-video w-full overflow-hidden rounded-lg">
                                <Image
                                    src={previewUrl}
                                    alt="Preview"
                                    fill
                                    className="object-cover"
                                />
                            </div>
                        ) : (
                            <div className="flex flex-col items-center gap-2 text-stone-500">
                                <ImageIcon className="h-10 w-10 text-stone-400" />
                                <span>Drag and drop or click to upload</span>
                            </div>
                        )}
                        <input
                            type="file"
                            accept="image/*"
                            className="absolute inset-0 cursor-pointer opacity-0"
                            onChange={handleFileChange}
                        />
                    </div>

                    <Button
                        className="w-full bg-amber-700 hover:bg-amber-800"
                        size="lg"
                        disabled={!selectedImage || loading}
                        onClick={handleClassify}
                    >
                        {loading ? (
                            <>
                                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                Analyzing...
                            </>
                        ) : (
                            "Identify Structure"
                        )}
                    </Button>

                    {error && <p className="text-sm text-red-500">{error}</p>}
                </CardContent>
            </Card>

            {/* Result Section */}
            <Card className="h-full bg-stone-50/50">
                <CardHeader>
                    <CardTitle>Analysis Result</CardTitle>
                </CardHeader>
                <CardContent>
                    {result ? (
                        <div className="animate-in fade-in slide-in-from-bottom-4 space-y-6">
                            <div className="rounded-lg border border-green-200 bg-green-50 p-4 text-center">
                                <p className="text-sm text-green-600 font-medium">Structure Identified</p>
                                <h2 className="mt-1 text-3xl font-bold text-green-800">{result.predicted_class}</h2>
                            </div>

                            <div className="space-y-2">
                                <div className="flex justify-between text-sm">
                                    <span>Confidence Score</span>
                                    <span className="font-bold">{(result.confidence * 100).toFixed(1)}%</span>
                                </div>
                                <div className="h-2 w-full overflow-hidden rounded-full bg-stone-200">
                                    <div
                                        className="h-full bg-amber-600 transition-all duration-500"
                                        style={{ width: `${result.confidence * 100}%` }}
                                    />
                                </div>
                            </div>

                            <div className="rounded-md bg-blue-50 p-4 text-sm text-blue-700">
                                <p>💡 <strong>Did you know?</strong> This is a key historical landmark in Indonesia, often visited for its cultural significance.</p>
                            </div>
                        </div>
                    ) : (
                        <div className="flex h-40 items-center justify-center text-stone-400">
                            <p>Results will appear here...</p>
                        </div>
                    )}
                </CardContent>
            </Card>
        </div>
    )
}
