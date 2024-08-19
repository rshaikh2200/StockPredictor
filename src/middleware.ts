import { clerkMiddleware } from "@clerk/nextjs/server";

export default clerkMiddleware({
    publishableKey: 'pk_test_YW11c2VkLWRpbmdvLTUwLmNsZXJrLmFjY291bnRzLmRldiQ',
    CLERK_SECRET_KEY: 'sk_test_6E2BHwtnDVm0WQHROqKNrIAIr4Hl0JNjg8tqPwuOgK'
});

export const config = {
  matcher: [
    // Skip Next.js internals and all static files, unless found in search params
    '/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)',
    // Always run for API routes
    '/(api|trpc)(.*)',
  ],
};
