"use client"

import { useState } from "react"
import { Search, MapPin, Loader2, Star, Tag } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent } from "@/components/ui/card"
import { recommendPlaces, Recommendation } from "@/lib/api"

const POPULAR_PLACES = ["Monumen Nasional", "Candi Borobudur", "Kota Tua", "Candi Prambanan"]

export function TripRecommender() {
    const [placeName, setPlaceName] = useState("")
    const [recommendations, setRecommendations] = useState<Recommendation[]>([])
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)

    const handleRecommend = async () => {
        if (!placeName) return

        setLoading(true)
        setError(null)
        setRecommendations([]) // Clear previous
        try {
            const data = await recommendPlaces(placeName)
            setRecommendations(data.recommendations)
        } catch (err) {
            const message = err instanceof Error ? err.message : "Failed to get recommendations."
            setError(message)
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="space-y-8">
            <div className="mx-auto max-w-2xl space-y-4 text-center">
                <h2 className="text-2xl font-semibold text-stone-800">Discover New Places</h2>
                <div className="flex gap-2">
                    <Input
                        placeholder="I enjoyed visiting... (e.g. Monas)"
                        value={placeName}
                        onChange={(e) => setPlaceName(e.target.value)}
                        className="h-12 text-lg shadow-sm"
                    />
                    <Button
                        className="h-12 bg-amber-700 px-8 hover:bg-amber-800"
                        onClick={handleRecommend}
                        disabled={loading || !placeName}
                    >
                        {loading ? <Loader2 className="animate-spin" /> : <Search />}
                    </Button>
                </div>

                <div className="flex flex-wrap items-center justify-center gap-2">
                    <span className="text-sm text-stone-500">Try:</span>
                    {POPULAR_PLACES.map((place) => (
                        <button
                            key={place}
                            onClick={() => setPlaceName(place)}
                            className="rounded-full border border-stone-200 bg-white px-3 py-1 text-sm text-stone-600 transition-colors hover:border-amber-500 hover:text-amber-700"
                        >
                            {place}
                        </button>
                    ))}
                </div>
                {error && <p className="text-red-500">{error}</p>}
            </div>

            {recommendations.length > 0 && (
                <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3 animate-in slide-in-from-bottom-8 fade-in duration-500">
                    {recommendations.map((rec, i) => (
                        <Card key={i} className="overflow-hidden transition-all hover:shadow-md">
                            <div className="h-32 bg-stone-200 relative">
                                <div className="absolute inset-0 flex items-center justify-center text-stone-400">
                                    <span className="text-sm">Image Placeholder</span>
                                    {/* Ideally fetch real image from Google Places API */}
                                </div>
                                <div className="absolute right-2 top-2 rounded-full bg-white/90 px-2 py-0.5 text-xs font-bold text-amber-700 shadow-sm">
                                    {rec.Rating.toFixed(1)} <Star className="inline h-3 w-3 mb-0.5" />
                                </div>
                            </div>
                            <CardContent className="p-4">
                                <div className="mb-2 flex items-start justify-between">
                                    <div>
                                        <h3 className="line-clamp-1 font-bold text-stone-900">{rec.Place_Name}</h3>
                                        <p className="flex items-center text-xs text-stone-500">
                                            <MapPin className="mr-1 h-3 w-3" /> {rec.City}
                                        </p>
                                    </div>
                                </div>

                                <div className="mt-4 flex items-center justify-between text-sm">
                                    <span className="inline-flex items-center rounded-md bg-stone-100 px-2 py-1 text-xs font-medium text-stone-600">
                                        <Tag className="mr-1 h-3 w-3" /> {rec.Category}
                                    </span>
                                    <span className="font-semibold text-stone-700">
                                        Rp {rec.Price.toLocaleString()}
                                    </span>
                                </div>

                                <div className="mt-3 text-xs text-stone-400">
                                    Match Score: {(rec.Similarity * 100).toFixed(0)}%
                                </div>
                            </CardContent>
                        </Card>
                    ))}
                </div>
            )}
        </div>
    )
}
