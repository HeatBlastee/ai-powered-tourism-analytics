import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { HeritageClassifier } from "@/components/HeritageClassifier"
import { TripRecommender } from "@/components/TripRecommender"

export default function Home() {
    return (
        <main className="min-h-screen bg-stone-50 text-stone-900 font-sans">
            {/* Header */}
            <header className="border-b border-stone-200 bg-white px-6 py-4 shadow-sm">
                <div className="mx-auto max-w-5xl flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <span className="text-2xl">🏛️</span>
                        <div>
                            <h1 className="text-xl font-bold tracking-tight text-stone-900">AI Tourism Analytics</h1>
                            <p className="text-xs text-stone-500">Preserving Heritage & Enhancing Travel</p>
                        </div>
                    </div>
                    <div className="text-sm font-medium text-green-600 bg-green-50 px-3 py-1 rounded-full">
                        ● API Connected
                    </div>
                </div>
            </header>

            {/* Main Content */}
            <div className="container mx-auto max-w-5xl py-12 px-4">
                <Tabs defaultValue="vision" className="space-y-8">
                    <div className="flex justify-center">
                        <TabsList className="grid w-full max-w-md grid-cols-2 bg-stone-200 text-stone-600">
                            <TabsTrigger value="vision">📷 Identify Heritage</TabsTrigger>
                            <TabsTrigger value="recommender">🗺️ Plan Trip</TabsTrigger>
                        </TabsList>
                    </div>

                    <TabsContent value="vision" className="animate-in fade-in-50 duration-500">
                        <div className="mx-auto max-w-3xl">
                            <div className="mb-8 text-center">
                                <h2 className="text-3xl font-bold text-stone-800">Heritage Structrue Identification</h2>
                                <p className="mt-2 text-stone-500">
                                    Upload a photo of a historical monument to identify its name and cultural significance.
                                </p>
                            </div>
                            <HeritageClassifier />
                        </div>
                    </TabsContent>

                    <TabsContent value="recommender" className="animate-in fade-in-50 duration-500">
                        <TripRecommender />
                    </TabsContent>
                </Tabs>
            </div>

            <footer className="mt-20 border-t border-stone-200 py-8 text-center text-stone-500 text-sm">
                <p>Built with Next.js, FastAPI & TensorFlow</p>
            </footer>
        </main>
    )
}
