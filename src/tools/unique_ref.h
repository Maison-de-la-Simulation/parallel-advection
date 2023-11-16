#pragma once

#include <functional>
#include <memory>
#include <stdexcept>

namespace sref {

template <class T, class Deleter = std::default_delete<T>>
class unique_ref
{
private:
    template <class U, class ODeleter>
    friend class unique_ref;

    std::unique_ptr<T> m_ptr;

public:
    using type = T;

    explicit unique_ref(std::unique_ptr<T> ptr) : m_ptr(std::move(ptr))
    {
        if (!m_ptr) {
            throw std::invalid_argument("nullptr");
        }
    }

    template <class U, class ODeleter>
    unique_ref(unique_ref<U, ODeleter> ref) noexcept : m_ptr(std::move(ref.m_ptr))
    {
    }

    unique_ref(unique_ref const& rhs) = default;

    unique_ref(unique_ref&& rhs) noexcept = default;

    ~unique_ref() noexcept = default;

    unique_ref& operator=(unique_ref const& rhs) = default;

    unique_ref& operator=(unique_ref&& rhs) noexcept = default;

    [[nodiscard]] explicit operator std::unique_ptr<T>() && noexcept
    {
        return std::move(m_ptr);
    }

    T& operator*() const noexcept
    {
        return *m_ptr;
    }

    T* operator->() const noexcept
    {
        return m_ptr.get();
    }

    template <class... Args>
    constexpr std::invoke_result_t<T&, Args...> operator()(Args&&... args) const
    {
        return std::invoke(*m_ptr, std::forward<Args>(args)...);
    }
};

template <class T, class Deleter>
unique_ref<T, Deleter> as_ref(std::unique_ptr<T, Deleter> ptr) noexcept
{
    return unique_ref<T, Deleter>(std::move(ptr));
}

template <class T, class Deleter>
unique_ref<std::add_const_t<T>, Deleter> as_cref(std::unique_ptr<T, Deleter> ptr) noexcept
{
    return unique_ref<std::add_const_t<T>, Deleter>(std::move(ptr));
}

template <class T, class... Args>
unique_ref<T> make_unique(Args&&... args)
{
    return as_ref(std::make_unique<T>(std::forward<Args>(args)...));
}

} // namespace sref
